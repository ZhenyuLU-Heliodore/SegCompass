import os, random, json, math
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from torch.utils.data import DistributedSampler

import numpy as np
import argparse
import torch
import torch.distributed as dist
import re
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    LlavaForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
)
from sae_harness.sae_dataset import build_mixed_dataset, SaeCacheActDataset
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.rl_dataset import collate_fn


def init_dist():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


def set_seeds_and_env(seed: int = 0, support_bf16: bool = True):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not support_bf16:
        os.environ["VLLM_ATTENTION_BACKEND"] = "SDPA"
        os.environ["VLLM_USE_TRITON"] = "0"
        os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_llm(args, device):
    args.llm_version = args.llm_version.strip().lower()
    llm_dtype = PrecisionType.to_dtype("bf16", args.support_bf16)

    if args.support_bf16:
        attn_impl = "flash_attention_2"
    else:
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True)
        attn_impl = "sdpa"

    if args.llm_version in ["qwen-2.5"]:
        model = init_qwen2_5(args, llm_dtype, attn_impl, device)
    elif args.llm_version in ["llava-1.5"]:
        model = init_llava(args, llm_dtype, attn_impl, device)
    else:
        raise ValueError(f"Unknown llm_version: {args.llm_version}.")

    model.eval()
    processor = AutoProcessor.from_pretrained(args.llm_hf)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.llm_hf, use_fast=False)
    return model, tokenizer, processor


def init_qwen2_5(args, llm_dtype, attn_impl, device):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.llm_hf,
        torch_dtype=llm_dtype,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    ).to(device)
    return model


def init_llava(args, llm_dtype, attn_impl, device):
    model = LlavaForConditionalGeneration.from_pretrained(
        args.llm_hf,
        torch_dtype=llm_dtype,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    ).to(device)
    return model


def forward_and_collect_hidden(args, model, batch: Dict, device):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    position_ids = batch["position_ids"].to(device, non_blocking=True)
    if position_ids.dim() == 3:  # qwen2vl mrope
        position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

    pixel_values = batch["pixel_values"]
    image_grid_thw = batch["image_grid_thw"]

    vision_inputs = {}
    if pixel_values and pixel_values[0] is not None:
        if "qwen" in args.llm_version:
            vision_inputs["image_grid_thw"] = torch.cat(image_grid_thw, dim=0).to(device, non_blocking=True)
        if args.llm_version in ["qwen-2.5"]:  # b * [l, d] _> [bl, d]
            vision_inputs["pixel_values"] = torch.cat(pixel_values, dim=0).to(device, non_blocking=True)
        elif args.llm_version in ["llava-1.5"]:  # b * [1, c, h, w] -> [b, c, h, w]
            vision_inputs["pixel_values"] = torch.cat(pixel_values, dim=0).to(device, non_blocking=True)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **vision_inputs,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True
        )

    hidden_states = outputs.hidden_states
    return hidden_states[int(args.sae_layer_k)]


def create_hex_buckets(hidden_save_dir):
    """create b00/ - bff/, and others/ """
    base = Path(hidden_save_dir)
    base.mkdir(parents=True, exist_ok=True)
    for i in range(256):
        (base / f"b{i:02x}").mkdir(exist_ok=True)
    (base / "others").mkdir(exist_ok=True)


def main(args):
    rank, world_size, local_rank = init_dist()
    device = torch.device(f"cuda:{local_rank}")
    create_hex_buckets(args.hidden_save_dir)
    llm_model, tokenizer, processor = init_llm(args, device)

    img_txt_dataset = build_mixed_dataset(
        args.data_dirs, args.image_dirs, args.splits
    )
    print(f"image-text dataset length: {len(img_txt_dataset)}")
    dataset = SaeCacheActDataset(
        img_txt_ds=img_txt_dataset,
        tokenizer=tokenizer,
        processor=processor,
        llm_version=args.llm_version
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=DistributedSampler(dataset, shuffle=False, drop_last=False),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    if rank == 0:
        print(f"#samples={len(dataset)}  #batches≈{len(dataloader)}")

    llm_model.eval()
    iterator = dataloader if rank != 0 else tqdm(dataloader, desc="caching activations")
    for batch in iterator:
        save_paths = [Path(args.hidden_save_dir) / p for p in batch["hidden_path"]]
        if all(p.exists() for p in save_paths):
            continue

        hidden = forward_and_collect_hidden(args, llm_model, batch, device).to("cpu", non_blocking=True)

        for i, save_path in enumerate(save_paths):
            hs_np = hidden[i].detach().cpu().to(torch.float16).numpy()
            am_np = batch["attention_mask"][i].detach().cpu().numpy()

            np.savez(save_path, hidden_states=hs_np, attention_mask=am_np)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--image_dirs", nargs="+", default=None)
    p.add_argument("--splits", nargs="+", default=None)
    p.add_argument("--llm_version")
    p.add_argument("--llm_hf", required=True)
    p.add_argument("--support_bf16", type=lambda s: False if s == "false" else True, default=True)
    p.add_argument("--sae_layer_k", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden_save_dir")

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
