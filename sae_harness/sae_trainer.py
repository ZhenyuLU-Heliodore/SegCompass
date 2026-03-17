import torch
import argparse
import re
import torch.nn.functional as F
import json
import numpy as np

from tqdm import tqdm
from sae_lens.sae import SAE, SAEConfig
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from omegaconf import OmegaConf
from sae_harness.cache_hiddens import init_dist
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


class SAETrainer:
    def __init__(self, config):
        self.config = config
        self.rank, self.world_size, self.local_rank = init_dist()
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.epoch = 0
        self.optimizer = None
        self.lr_scheduler = None

    def _build_sae(self, trainable=True):
        cfg_dict = OmegaConf.to_container(self.config.sae_model, resolve=True)
        sae_cfg = SAEConfig.from_dict(cfg_dict)
        self.sae = SAE(sae_cfg).to(self.device)
        for p in self.sae.parameters():
            p.requires_grad_(trainable)
        self.sae.train(trainable)
        self.sae = DDP(
            self.sae,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True
        )
        return self.sae

    def _build_dataloader(self):
        dataset = CachedActDataset(self.config.data.cached_dir)

        n_valid = int(len(dataset) * 0.05)
        n_train = len(dataset) - n_valid

        g = torch.Generator()
        g.manual_seed(int(self.config.data.split_seed))
        train_set, valid_set = random_split(dataset, [n_train, n_valid], generator=g)

        train_sampler = DistributedSampler(
            train_set, num_replicas=self.world_size, rank=self.rank,
            shuffle=True, drop_last=False
        )
        valid_sampler = DistributedSampler(
            valid_set, num_replicas=self.world_size, rank=self.rank,
            shuffle=False, drop_last=False
        )
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.config.data.batch_size,
            multiprocessing_context="spawn",
            sampler=train_sampler,
            num_workers=4,
            prefetch_factor=1,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=_collate,
            drop_last=False,
        )
        self.valid_loader = DataLoader(
            valid_set,
            batch_size=self.config.data.batch_size,
            multiprocessing_context="spawn",
            sampler=valid_sampler,
            num_workers=4,
            prefetch_factor=1,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=_collate,
            drop_last=False,
        )

        return self.train_loader, self.valid_loader

    def _build_optimizer(self):
        base_lr = float(self.config.optim.lr)
        self.optimizer = torch.optim.AdamW(self.sae.parameters(), lr=base_lr)

        total_steps = int(self.config.train.max_epochs) * len(self.train_loader) + 1  # avoid shift
        warmup_steps = int(total_steps * self.config.optim.warmup_ratio)
        cosine_steps = total_steps - warmup_steps

        cosine = CosineAnnealingLR(
            self.optimizer, T_max=cosine_steps, eta_min=base_lr * self.config.optim.eta_min_factor,
        )
        if warmup_steps > 0:
            warmup = LinearLR(
                self.optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps,
            )
            self.lr_scheduler = SequentialLR(self.optimizer, [warmup, cosine], [warmup_steps])
        else:
            self.lr_scheduler = cosine
        return self.optimizer, self.lr_scheduler

    def _compute_loss(self, batch):
        """
        batch should contain:
          - hidden_states: (B, L, d_in)
          - attention_mask: (B, L)
        """
        assert self.config.sae_model.architecture == "standard", "Only support standard arch here."
        x = batch["hidden_states"].to(device=self.device, dtype=next(self.sae.parameters()).dtype)
        mask = batch["attention_mask"].to(self.device)
        valid = (mask != 0).float()

        sparse_embeds, x_recon = self.sae(x)

        se_token = F.mse_loss(x_recon, x, reduction='none').sum(-1)  # [b, l, d_llm] -> [b, l]
        mse_loss = (se_token * valid).sum() / valid.sum().clamp_min(1)  # [b, l] -> value

        weighted_sparse_embeds = sparse_embeds * self.sae.module.W_dec.norm(dim=1)  # [b, l, d_sae] * [d_sae] -> [b, l, d_sae]
        sparsity = weighted_sparse_embeds.norm(p=1, dim=-1)  # [b, l, d_sae] -> [b, l]
        l1_loss = (sparsity * valid).sum() / valid.sum().clamp_min(1)  # [b, l] -> value
        loss = mse_loss + self.config.loss.l1_coef * l1_loss

        return mse_loss, l1_loss, loss

    def _save_checkpoint(self):
        if self.rank != 0:
            return
        save_dir = Path(self.config.ckpt.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"ep_{self.epoch}.pt"

        ckpt = {
            "sae": {k: v.cpu() for k, v in self.sae.module.state_dict().items()},
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "epoch": self.epoch,
        }
        torch.save(ckpt, path)
        return str(path)

    def _load_checkpoint(self, load_path):
        """Load from dir or file. If dir, find the latest"""
        if not load_path:
            return None

        load_path = Path(load_path)
        _EP_RE = re.compile(r"ep_(\d+)\.pt$")
        if load_path.is_dir():
            candidates = []
            for p in load_path.glob("ep_*.pt"):
                m = _EP_RE.search(p.name)
                if m:
                    candidates.append((int(m.group(1)), p))
            if not candidates:
                return None
            _, load_path = max(candidates, key=lambda t: t[0])

        ckpt = torch.load(load_path, map_location="cpu")
        self.sae.module.load_state_dict(ckpt["sae"])
        if ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt["scheduler"] is not None:
            self.lr_scheduler.load_state_dict(ckpt["scheduler"])

        self.epoch = int(ckpt.get("epoch"))
        return str(load_path)

    def _write_log(self, log, epoch=None, step=None):
        itr_name = "Step" if step is not None else "Epoch"
        itr_point = step if step is not None else epoch
        log_path = Path(self.config.ckpt.save_dir) / "log.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"======== {itr_name} {itr_point} ========\n")
            f.write(json.dumps(log, ensure_ascii=False, sort_keys=True, indent=2) + "\n\n")

    def _reduce_metrics(self, metrics):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
            metrics /= self.world_size
        return metrics

    def _train_loop(self, epoch):
        train_metrics = []
        self.sae.train()
        train_itr = self.train_loader if self.rank != 0 else tqdm(self.train_loader, desc=f"Train Epoch: {epoch}")

        for step, batch in enumerate(train_itr):
            self.optimizer.zero_grad()

            mse_loss, l1_loss, loss = self._compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)

            self.optimizer.step()
            self.lr_scheduler.step()

            metrics = torch.stack([mse_loss.detach(), l1_loss.detach(), loss.detach()]).to(self.device)
            metrics = self._reduce_metrics(metrics)
            metric = {
                "mse_loss": metrics[0].cpu().item(),
                "l1_loss": metrics[1].cpu().item(),
                "loss": metrics[2].cpu().item(),
            }
            train_metrics.append(metric)
            if self.rank == 0:
                global_step = step + epoch * len(train_itr)
                self._write_log(metric, step=global_step)

        return {k: sum(m[k] for m in train_metrics) / len(train_metrics) for k in train_metrics[0]}

    def _validate(self, epoch):
        self.sae.eval()
        valid_metrics = []
        valid_itr = self.valid_loader if self.rank != 0 else tqdm(self.valid_loader, desc=f"Valid Epoch: {epoch}")

        with torch.no_grad():
            for batch in valid_itr:
                mse_loss, l1_loss, loss = self._compute_loss(batch)
                metrics = torch.stack([mse_loss.detach(), l1_loss.detach(), loss.detach()]).to(self.device)
                metrics = self._reduce_metrics(metrics)

                valid_metrics.append({
                    "mse_loss": metrics[0].cpu().item(),
                    "l1_loss": metrics[1].cpu().item(),
                    "loss": metrics[2].cpu().item(),
                })
        log = {k: sum(m[k] for m in valid_metrics) / len(valid_metrics) for k in valid_metrics[0]}
        if self.rank == 0:
            self._write_log(log, epoch=epoch)
        return log

    def fit(self):
        self._build_dataloader()
        self._build_sae()
        self._build_optimizer()

        self._load_checkpoint(self.config.ckpt.load_path)
        if Path(self.config.ckpt.save_dir).is_dir():
            self._load_checkpoint(self.config.ckpt.save_dir)  # load the latest ckpt

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.config.train.max_epochs):
            self.epoch = epoch
            self.train_loader.sampler.set_epoch(epoch)
            _ = self._train_loop(epoch)
            _ = self._validate(epoch)

            if self.rank == 0:
                self._save_checkpoint()

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


class CachedActDataset(Dataset):
    def __init__(self, cached_dir):
        super().__init__()
        self.cached_dir = Path(cached_dir)
        self._cached_files = sorted(
            p for p in self.cached_dir.rglob("*.npz") if p.is_file() and p.name != "zero_sample.npz"
        )

        with np.load(self.cached_dir / "zero_sample.npz") as npz:
            hs = torch.as_tensor(npz["hidden_states"], dtype=torch.bfloat16)
            am = torch.as_tensor(npz["attention_mask"], dtype=torch.int64)
        self.zero_sample = {"hidden_states": hs, "attention_mask": am}

    def __len__(self):
        return len(self._cached_files)

    def __getitem__(self, idx):
        path = self._cached_files[idx]
        try:
            with np.load(path) as npz:
                hs = torch.as_tensor(npz["hidden_states"], dtype=torch.bfloat16)
                am = torch.as_tensor(npz["attention_mask"], dtype=torch.int64)
            return {"hidden_states": hs, "attention_mask": am}

        except Exception:
            print(f"[warning] bad sample -> {path}, fallback zero_sample")
            return self.zero_sample.copy()


def _collate(batch):
    return {
        "hidden_states": torch.stack([b["hidden_states"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
    }


def load_cfg():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-c", "--config", required=True)
    args, rest = p.parse_known_args()

    yaml_cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_cli(rest)
    return OmegaConf.merge(yaml_cfg, cli_cfg)


def main():
    config = load_cfg()
    trainer = SAETrainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()
