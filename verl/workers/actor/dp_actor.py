# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement Actor
"""

import os
import json
import numpy as np
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer import core_algos
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.actor.base import BasePPOActor
from verl.workers.actor.config import ActorConfig
from verl.utils.rl_dataset import clamp_llava_image_tokens

__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
            self,
            config: ActorConfig,
            actor_module: nn.Module,
            actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)
        self.global_steps = 0

    def _forward_micro_batch(
            self, micro_batch: Dict[str, torch.Tensor], meta_info: Dict = None, forward_seg: bool = False,
    ) -> Dict:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            seg_outputs: ...
        """
        temperature = meta_info["temperature"]
        llm_version, image_token_id, replace_token_id = (
            meta_info["llm_version"], meta_info["image_token_id"], meta_info["replace_token_id"])
        input_ids = micro_batch["input_ids"] if "llava" not in llm_version else \
            clamp_llava_image_tokens(micro_batch["input_ids"], image_token_id, replace_token_id)

        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        image_embed = micro_batch["sam_embed"] if forward_seg else None
        unpadded_hw = micro_batch["unpadded_hw"] if forward_seg else None
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        llm_inputs = {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "position_ids": position_ids, "use_cache": False,
        }

        sae_llm_inputs = None
        if forward_seg:
            sae_input_ids = micro_batch["sae_input_ids"] if "llava" not in llm_version else \
                clamp_llava_image_tokens(micro_batch["sae_input_ids"], image_token_id, replace_token_id)

            sae_attention_mask = micro_batch["sae_attention_mask"]
            sae_position_ids = micro_batch["sae_position_ids"]
            if sae_position_ids.dim() == 3:  # qwen2vl mrope
                sae_position_ids = sae_position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
            sae_llm_inputs = {
                "input_ids": sae_input_ids, "attention_mask": sae_attention_mask,
                "position_ids": sae_position_ids, "use_cache": False,
            }

        if "pixel_values" in micro_batch:
            pixel_values = torch.cat(micro_batch["pixel_values"], dim=0)
            image_grid_thw = torch.cat(micro_batch["image_grid_thw"], dim=0) \
                if micro_batch["image_grid_thw"][0] is not None else None
            llm_inputs["pixel_values"] = pixel_values
            llm_inputs["image_grid_thw"] = image_grid_thw
            if forward_seg:
                sae_llm_inputs["pixel_values"] = pixel_values
                sae_llm_inputs["image_grid_thw"] = image_grid_thw

        output = self.actor_module(
            forward_seg=forward_seg,
            image_embed=image_embed,
            unpadded_hw=unpadded_hw,
            **llm_inputs,
            sae_llm_inputs=sae_llm_inputs,
        )
        logits: torch.Tensor = output.logits
        logits.div_(temperature)
        logits = logits[:, -response_length - 1: -1, :]  # (bsz, response_length, vocab_size)
        log_probs = logprobs_from_logits(logits, responses)  # (bsz, response_length)
        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

        outputs = {"entropy": entropy, "log_probs": log_probs}
        if forward_seg:
            outputs["seg_outputs"] = output.seg_outputs

        return outputs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        self.actor_optimizer.step()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto, forward_seg: bool = False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys
                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.
                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.
                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.
                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.
                ``sam_embed``: optional, if gen_mask = True, required.

            forward_seg: whether to generate segmentation mask for computing reward function

        Returns:
            if gen_mask is False:
                torch.Tensor: the log_prob tensor
            if gen_mask is True:
                torch.Tensor: the log_prob tensor
                torch.Tensor: mask_sigmoid_detach
                torch.Tensor: conf_logits_detach

        """
        self.actor_module.eval()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", ]
        if forward_seg:
            select_keys += ["sam_embed", "unpadded_hw", "sae_input_ids", "sae_attention_mask", "sae_position_ids"]

        if "pixel_values" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["pixel_values", "image_grid_thw"]
        else:
            non_tensor_select_keys = None

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst, mask_sigmoid_lst, conf_logits_lst, = [], [], []
        for micro_batch in tqdm(micro_batches, desc="Compute log probs", disable=(self.rank != 0)):
            micro_batch.to("cuda")
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            forward_outputs = self._forward_micro_batch(
                model_inputs, meta_info=data.meta_info, forward_seg=forward_seg
            )

            log_probs_lst.append(forward_outputs["log_probs"])
            if forward_seg:
                seg_outputs = forward_outputs["seg_outputs"]
                mask_sigmoid_lst.append(seg_outputs["mask_sigmoid"])
                conf_logits_lst.append(seg_outputs["conf_logits"])

        log_probs = torch.concat(log_probs_lst, dim=0)

        if not forward_seg:
            return log_probs

        return log_probs, torch.concat(mask_sigmoid_lst, dim=0), torch.concat(conf_logits_lst, dim=0)

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages",
                       "sam_embed", "n_multi_objects", "mask_float_256", "unpadded_hw",
                       "sae_input_ids", "sae_attention_mask", "sae_position_ids", ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        if "pixel_values" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["pixel_values", "image_grid_thw"]
        else:
            non_tensor_select_keys = None

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        n = len(mini_batches)
        for i, mini_batch in enumerate(mini_batches):
            gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
            )
            micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

            self.actor_optimizer.zero_grad()

            for micro_batch in tqdm(micro_batches, desc=f"Update policy [{i + 1}/{n}]", disable=(self.rank != 0)):
                micro_batch.to("cuda")
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                responses = model_inputs["responses"]
                response_length = responses.size(1)
                attention_mask = model_inputs["attention_mask"]
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = model_inputs["old_log_probs"]
                advantages = model_inputs["advantages"]

                clip_ratio = self.config.clip_ratio
                entropy_coef = self.config.entropy_coef

                # all return
                outputs = self._forward_micro_batch(
                    model_inputs, meta_info=data.meta_info, forward_seg=True
                )
                entropy, log_prob, seg_outputs = outputs["entropy"], outputs["log_probs"], outputs["seg_outputs"]

                # compute seg loss
                if self.config.model.k_slots > 1:
                    seg_loss, heatmap_bce, dice_loss, focal_loss, conf_loss = (
                        self._multiple_seg_loss(seg_outputs, model_inputs))
                else:
                    seg_loss, heatmap_bce, dice_loss, focal_loss = (
                        self._single_seg_loss(seg_outputs, model_inputs))

                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                pg_loss, pg_clipfrac, ppo_kl, ppo_ratio = core_algos.compute_policy_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    eos_mask=response_mask,
                    cliprange=clip_ratio,
                )
                policy_loss = pg_loss - entropy_loss * entropy_coef

                if self.config.use_kl_loss:
                    ref_log_prob = model_inputs["ref_log_prob"]
                    # compute kl loss
                    kld = core_algos.kl_penalty(
                        logprob=log_prob,
                        ref_logprob=ref_log_prob,
                        kl_penalty=self.config.kl_loss_type,
                    )
                    kl_loss = verl_F.masked_mean(kld, response_mask)
                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef

                loss = (self.config.pg_loss_coef * policy_loss + self.config.seg_loss_coef * seg_loss)
                if self.config.model.k_slots > 1:
                    loss += self.config.conf_loss_coef * conf_loss
                loss = loss / gradient_accumulation

                loss.backward()

                batch_metrics = {
                    "policy_loss": policy_loss.detach().item(),
                    "seg_loss": seg_loss.detach().item(),
                    "heatmap_bce": heatmap_bce.detach().item(),
                    "dice_loss": dice_loss.detach().item(),
                    "focal_loss": focal_loss.detach().item(),
                    "conf_loss": conf_loss.detach().item() if self.config.model.k_slots > 1 else 0.0,
                    "ratio_max": ppo_ratio.detach().max().item(),
                    "pg_abs": pg_loss.abs().detach().item(),
                    "kl_loss": kl_loss.detach().item() if self.config.use_kl_loss else 0.0,
                    "pg_clipfrac": pg_clipfrac.detach().item(),
                }
                append_to_dict(metrics, batch_metrics)

            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {"grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics

    def _single_seg_loss(self, seg_outputs, model_inputs):
        ref_logits_256, gt_mask_256 = seg_outputs["ref_logits_256"], model_inputs["mask_float_256"]
        mask_logits, mask_sigmoid = seg_outputs["mask_logits"], seg_outputs["mask_sigmoid"]
        gt_mask_256 = gt_mask_256.to(device=ref_logits_256.device, dtype=ref_logits_256.dtype)
        dev = ref_logits_256.device
        up_hw = torch.as_tensor(model_inputs["unpadded_hw"], device=dev, dtype=torch.long)  # (B,2)
        if up_hw.ndim != 2 or up_hw.size(-1) != 2:
            raise ValueError(f"'unpadded_hw' shape must be (B,2), got {tuple(up_hw.shape)}")

        B, _, H, W = ref_logits_256.shape

        # 1024 → 256：(x + 3) // 4
        h256 = ((up_hw[:, 0].clamp(min=0) + 3) // 4).clamp(max=H)  # (B,)
        w256 = ((up_hw[:, 1].clamp(min=0) + 3) // 4).clamp(max=W)  # (B,)

        yy = torch.arange(H, device=dev).view(1, H, 1)  # (1,H,1)
        xx = torch.arange(W, device=dev).view(1, 1, W)  # (1,1,W)
        valid_256 = (yy < h256.view(B, 1, 1)) & (xx < w256.view(B, 1, 1))  # (B,H,W) bool
        valid_256 = valid_256.float().unsqueeze(1)  # (B,1,H,W)

        bce_map = F.binary_cross_entropy_with_logits(ref_logits_256, gt_mask_256, reduction="none")
        heatmap_bce = (bce_map * valid_256).sum() / (valid_256.sum() + 1e-6)
        focal_map = sigmoid_focal_loss(mask_logits, gt_mask_256, reduction="none")
        focal_loss = (focal_map * valid_256).sum() / (valid_256.sum() + 1e-6)

        heatmap_dice_loss = 1.0 - soft_dice(
            torch.sigmoid(ref_logits_256), gt_mask_256, valid=valid_256, reduction="mean"
        )
        dice_loss = 1.0 - soft_dice(mask_sigmoid, gt_mask_256, valid=valid_256, reduction="mean")

        if self.config.adjust_loss_step < 0 or self.global_steps < self.config.adjust_loss_step:
            dice_coef, focal_coef = self.config.dice_loss_coef, self.config.focal_loss_coef
        else:
            dice_coef, focal_coef = self.config.dice_loss_coef_new, self.config.focal_loss_coef_new
        heatmap_dice_coef = dice_coef * self.config.heatmap_dice_rate
        seg_loss = (heatmap_bce
                    + dice_coef * dice_loss
                    + focal_coef * focal_loss
                    + heatmap_dice_loss * heatmap_dice_coef)
        return seg_loss, heatmap_bce, dice_loss, focal_loss

    def _multiple_seg_loss(self, seg_outputs, model_inputs):
        ref_logits_256, conf_logits = seg_outputs["ref_logits_256"], seg_outputs["conf_logits"]
        mask_logits, mask_sigmoid = seg_outputs["mask_logits"], seg_outputs["mask_sigmoid"]
        n_multi_objects, gt_mask_256 = model_inputs["n_multi_objects"], model_inputs["mask_float_256"]
        up_hw = model_inputs["unpadded_hw"]

        B, K, H, W = ref_logits_256.shape
        dev = ref_logits_256.device
        up_hw = torch.as_tensor(up_hw, device=dev, dtype=torch.long)

        h256 = ((up_hw[:, 0].clamp(min=0) + 3) // 4).clamp(max=H)
        w256 = ((up_hw[:, 1].clamp(min=0) + 3) // 4).clamp(max=W)
        yy = torch.arange(H, device=dev).view(H, 1)
        xx = torch.arange(W, device=dev).view(1, W)

        heatmap_bce_lst, heatmap_dice_lst, dice_lst, focal_lst = [], [], [], []
        conf_targets = []
        for b in range(B):
            n_gt = int(n_multi_objects[b].item())
            vmask = ((yy < h256[b].item()) & (xx < w256[b].item())).float()  # [H,W]
            denom = vmask.sum() + 1e-6

            if n_gt == 0:
                conf_targets.append(torch.zeros(K, device=dev, dtype=conf_logits.dtype))
                heatmap_bce_lst.append(ref_logits_256.new_zeros(()))
                heatmap_dice_lst.append(ref_logits_256.new_zeros(()))
                dice_lst.append(ref_logits_256.new_zeros(()))
                focal_lst.append(ref_logits_256.new_zeros(()))
                continue

            pred_prob = torch.sigmoid(ref_logits_256[b])  # [K,H,W]
            gt_b = gt_mask_256[b, :n_gt]  # [G,H,W]

            iou = verl_F.pairwise_soft_iou(pred_prob, gt_b)  # [K,G]
            r, c = linear_sum_assignment((1.0 - iou).detach().cpu().numpy())
            r = torch.as_tensor(r, device=dev)
            c = torch.as_tensor(c, device=dev)
            m = r.numel()

            tgt = torch.zeros(K, device=dev, dtype=conf_logits.dtype)
            tgt.index_fill_(0, r, 1.0)
            conf_targets.append(tgt)

            pred_logit_m = ref_logits_256[b, r]  # [m,H,W]
            pred_prob_m = torch.sigmoid(pred_logit_m)  # [m,H,W]
            mask_logit_m = mask_logits[b, r]  # [m,H,W]
            mask_prob_m = mask_sigmoid[b, r]  # [m,H,W]
            gt_m = gt_b[c]  # [m,H,W]

            v = vmask.unsqueeze(0)  # [1,H,W]
            hb = (F.binary_cross_entropy_with_logits(pred_logit_m, gt_m, reduction="none") * v).flatten(1).sum(
                1) / denom
            heatmap_bce_lst.append(hb.mean())

            v_exp = v.unsqueeze(1).expand(m, 1, H, W)  # [m,1,H,W]
            hd = 1.0 - soft_dice(pred_prob_m.unsqueeze(1), gt_m.unsqueeze(1), valid=v_exp, reduction="mean")
            heatmap_dice_lst.append(hd)

            foc_map = sigmoid_focal_loss(mask_logit_m, gt_m, reduction="none")
            fl = (foc_map * v).flatten(1).sum(1) / denom
            focal_lst.append(fl.mean())

            dl = 1.0 - soft_dice(mask_prob_m.unsqueeze(1), gt_m.unsqueeze(1), valid=v_exp, reduction="mean")
            dice_lst.append(dl)

        conf_target = torch.stack(conf_targets, dim=0)  # [B,K]
        conf_loss = F.binary_cross_entropy_with_logits(conf_logits, conf_target, reduction="mean")

        heatmap_bce = torch.stack(heatmap_bce_lst).mean()
        heatmap_dice_loss = torch.stack(heatmap_dice_lst).mean()
        dice_loss = torch.stack(dice_lst).mean()
        focal_loss = torch.stack(focal_lst).mean()

        if self.config.adjust_loss_step < 0 or self.global_steps < self.config.adjust_loss_step:
            dice_coef, focal_coef = self.config.dice_loss_coef, self.config.focal_loss_coef
        else:
            dice_coef, focal_coef = self.config.dice_loss_coef_new, self.config.focal_loss_coef_new

        heatmap_dice_coef = dice_coef * self.config.heatmap_dice_rate
        seg_loss = (heatmap_bce
                    + dice_coef * dice_loss
                    + focal_coef * focal_loss
                    + heatmap_dice_loss * heatmap_dice_coef
                    )

        return seg_loss, heatmap_bce, dice_loss, focal_loss, conf_loss

    def evaluate(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.eval()
        save_metric_path = os.path.join(data.meta_info["write_eval_dir"], "rank_"+str(self.rank)+".txt")

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "mask_bool_gt_padded",
                       "image_id", "sam_embed", "n_multi_objects", "unpadded_hw", "original_hw",
                       "sae_input_ids", "sae_attention_mask", "sae_position_ids", ]

        non_tensor_select_keys = ["pixel_values", "image_grid_thw"]

        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)
        eval_metrics = defaultdict(list)
        inters, unions = 0.0, 0.0

        for i, mini_batch in enumerate(mini_batches):
            micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
            for micro_batch in micro_batches:
                micro_batch.to("cuda")
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                forward_outputs = self._forward_micro_batch(
                    model_inputs, meta_info=data.meta_info, forward_seg=True
                )
                seg_outputs = forward_outputs["seg_outputs"]

                ref_logits_256 = seg_outputs["ref_logits_256"]  # (B, K, 256, 256)
                mask_logits_256 = seg_outputs["mask_logits"]  # (B, K, 256, 256)
                conf_logits = seg_outputs["conf_logits"]  # (B, K)

                image_ids = model_inputs["image_id"]
                up_hws = model_inputs["unpadded_hw"]
                orig_hws = model_inputs["original_hw"]
                n_objs = model_inputs["n_multi_objects"]
                gt_padded = model_inputs["mask_bool_gt_padded"].to(device=mask_logits_256.device,
                                                                   dtype=torch.bool)  # (B,K,*,*)

                B, K, _, _ = ref_logits_256.shape
                for b in range(B):
                    img_id = int(image_ids[b].item())
                    n_gt = int(n_objs[b].item())
                    if n_gt == 0:
                        continue

                    up_h, up_w = int(up_hws[b][0].item()), int(up_hws[b][1].item())
                    orig_h, orig_w = int(orig_hws[b][0].item()), int(orig_hws[b][1].item())

                    heat_1024 = F.interpolate(ref_logits_256[b].unsqueeze(1), (1024, 1024), mode="bilinear",
                                              align_corners=False).squeeze(1)  # (K,1024,1024)
                    mask_1024 = F.interpolate(mask_logits_256[b].unsqueeze(1), (1024, 1024), mode="bilinear",
                                              align_corners=False).squeeze(1)  # (K,1024,1024)
                    heat_up = heat_1024[..., :up_h, :up_w]
                    mask_up = mask_1024[..., :up_h, :up_w]
                    heat_orig = F.interpolate(heat_up.unsqueeze(1), (orig_h, orig_w), mode="bilinear",
                                              align_corners=False).squeeze(1)  # (K,H,W)
                    mask_orig = F.interpolate(mask_up.unsqueeze(1), (orig_h, orig_w), mode="bilinear",
                                              align_corners=False).squeeze(1)  # (K,H,W)

                    heat_bool = (heat_orig > 0.0)
                    mask_bool = (mask_orig > 0.0)

                    gt = gt_padded[b, :n_gt, :orig_h, :orig_w]  # (G,H,W)

                    with torch.no_grad():
                        hb = heat_bool.unsqueeze(1)  # (K,1,H,W)
                        gb = gt.unsqueeze(0)  # (1,G,H,W)
                        inter_h = (hb & gb).sum(dim=(-2, -1)).float()
                        union_h = (hb | gb).sum(dim=(-2, -1)).float() + 1e-6
                        iou_heat = (inter_h / union_h)  # (K,G)

                        mb = mask_bool.unsqueeze(1)
                        inter_m = (mb & gb).sum(dim=(-2, -1)).float()
                        union_m = (mb | gb).sum(dim=(-2, -1)).float() + 1e-6
                        iou_mask = (inter_m / union_m)  # (K,G)

                    r, c = linear_sum_assignment((1.0 - iou_heat).detach().cpu().numpy())
                    r = torch.as_tensor(r, device=iou_heat.device)
                    c = torch.as_tensor(c, device=iou_heat.device)
                    m = r.numel()

                    vals_heat = iou_heat[r, c]
                    vals_mask = iou_mask[r, c]
                    eval_metrics[img_id].append([float(vals_mask.mean().item()), float(vals_heat.mean().item())])

                    inters += float(inter_m[r, c].sum().item())
                    unions += float(union_m[r, c].sum().item())

        open(save_metric_path, "a").write(json.dumps(eval_metrics, ensure_ascii=False,
                                                     default=lambda o: o.item() if hasattr(o, "item") else (
                                                         o.tolist() if hasattr(o, "tolist") else o)) + "\n")
        return eval_metrics


def soft_dice(
        prob: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
        eps: float = 1e-6,
        reduction: str = "mean",
) -> torch.Tensor:
    """Compute soft Dice score over (B, C, H, W) probabilities and targets, then mean-reduce over batch."""
    target = target.to(dtype=prob.dtype, device=prob.device)

    # --- minimal change: apply valid mask if provided ---
    if valid is not None:
        valid = valid.to(dtype=prob.dtype, device=prob.device)
        if valid.ndim == prob.ndim - 1:  # allow (B,H,W)
            valid = valid.unsqueeze(1)
        prob = prob * valid
        target = target * valid

    # sum over channel/spatial dims, keep batch
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    union = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter) / (union + eps)

    if reduction == "mean":
        return dice.mean()
    if reduction == "sum":
        return dice.sum()
    return dice

