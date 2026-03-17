import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass
from typing import Optional, Dict
from segment_anything import sam_model_registry
from transformers.utils import ModelOutput
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
)
from sae_lens.sae import SAE, SAEConfig


@dataclass
class Qwen2_5_VLWithSegOutput(Qwen2_5_VLCausalLMOutputWithPast):
    seg_outputs: Optional[Dict[str, torch.Tensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None  # For consistence with llava


class VLRefSegCore(nn.Module):
    """
    VL Segmentation Core.
    forward(...):
      Inputs
        - image_embed: (B, 256, 64, 64)
        - unpadded_hw: (B, 2), resized image size with the longer side being fixed
        - sae_llm_inputs: dict (same keys as llm_inputs; second LLM pass for SAE)
        - **llm_inputs: must include 'input_ids' (and usually 'attention_mask')

      Returns
        - Same LLM ModelOutput type.
        - If forward_seg=True, adds seg_outputs:
            * ref_logits_256: (B, K, 256, 256)   # from heatmap head
            * conf_logits:    (B, K)             # from heatmap head
            * mask_logits:    (B, K, 256, 256)   # from SAM decoder
            * mask_sigmoid:   (B, K, 256, 256)
            * ref_vec:        (B, K, D_llm)
    """
    def __init__(
        self,
        llm: nn.Module,
        sae: SAE,
        query_book: nn.Module,
        heatmap_head: nn.Module,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        sae_hooked_layer: int,
        special_token_id: int = None,
    ):
        super().__init__()
        self.llm = llm
        self.sae = sae
        self.query_book = query_book
        self.heatmap_head = heatmap_head
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.sae_hooked_layer = sae_hooked_layer

        # SAFETY: special_token_id must be provided
        if special_token_id is None:
            raise ValueError("special_token_id must not be None. Pass tokenizer.convert_tokens_to_ids('<REF_POS>')")
        # Keep on-module device; not saved in checkpoint (persistent=False).
        self.register_buffer("ref_pos_id", torch.tensor(special_token_id, dtype=torch.long), persistent=False)

    def get_ref_vec(self, llm_input_ids, last_hidden_state):
        # last K <REF_POS> per sample, pad zeros if fewer than K
        B, T = llm_input_ids.shape
        D = last_hidden_state.size(-1)
        device = llm_input_ids.device
        K = int(getattr(self.query_book, "k_slots", 1))

        mask = (llm_input_ids == self.ref_pos_id.to(device))  # [B,T]
        tidx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B,T]
        scores = torch.where(mask, tidx, torch.full_like(tidx, -1))  # -1 for non-matches
        vals, idx = scores.topk(K, dim=-1, largest=True, sorted=True)  # [B,K] latest→earlier

        valid = (vals >= 0).unsqueeze(-1)  # [B,K,1]
        safe_idx = idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, D)  # [B,K,D]
        ref_vec = last_hidden_state.to(device).gather(1, safe_idx)  # [B,K,D]
        return ref_vec * valid  # zero-pad where missing

    def forward(
            self,
            forward_seg: bool = True,
            image_embed: torch.Tensor = None,
            unpadded_hw: torch.Tensor = None,
            sae_llm_inputs: Dict[str, torch.Tensor] = None,
            **llm_inputs,
    ) -> ModelOutput:

        # Use no cache so hidden states align with full sequence (esp. training).
        llm_inputs.setdefault("use_cache", False)

        # Call the LLM. Note: do NOT forward image_embed unless model expects it in kwargs.
        # If your LLM expects visual inputs in llm_inputs (e.g., pixel_values), include them there before this call.
        llm_outputs = self.llm(output_hidden_states=forward_seg, **llm_inputs,)

        if not forward_seg:
            return llm_outputs

        # ======== Forward Segmentation Relative Modules ========
        ref_vec = self.get_ref_vec(
            llm_input_ids=llm_inputs["input_ids"],
            last_hidden_state=llm_outputs.hidden_states[-1]
        )

        sae_llm_outputs = self.llm(output_hidden_states=True, **sae_llm_inputs)

        sae_in = sae_llm_outputs.hidden_states[int(self.sae_hooked_layer)]
        sae_in = sae_in.to(device=next(self.sae.parameters()).device, dtype=next(self.sae.parameters()).dtype)
        sae_embeds, _ = self.sae(sae_in)

        seg_outputs = self._forward_seg(
            image_embed=image_embed,
            ref_vec=ref_vec,
            sae_embeds=sae_embeds,
            sae_attention_mask=sae_llm_inputs["attention_mask"],
            unpadded_hw=unpadded_hw,
        )
        llm_output_dict = dict(llm_outputs)

        return Qwen2_5_VLWithSegOutput(**llm_output_dict, seg_outputs=seg_outputs)

    def _forward_seg(self, image_embed, ref_vec, sae_embeds, sae_attention_mask, unpadded_hw):
        # ======== forward by QueryBookHead =========
        qb_dtype, qb_device = next(self.query_book.parameters()).dtype, next(self.query_book.parameters()).device
        querybook_slots = self.query_book(
            sae_embeds=sae_embeds.to(qb_device, qb_dtype),
            attention_mask=sae_attention_mask.to(qb_device)
        )

        # ========= forward by HeatmapHead =========
        hm_dtype, hm_device = next(self.heatmap_head.parameters()).dtype, next(self.heatmap_head.parameters()).device
        seg_outputs = self.heatmap_head(
            image_embed=image_embed.to(dtype=hm_dtype, device=hm_device),
            ref_vec=ref_vec.to(dtype=hm_dtype, device=hm_device),
            querybook_slots=querybook_slots.to(dtype=hm_dtype, device=hm_device),
        )

        pe_dtype, pe_device = next(self.prompt_encoder.parameters()).dtype, next(self.prompt_encoder.parameters()).device
        seg_outputs["ref_vec"] = ref_vec.to(device=pe_device, dtype=pe_dtype)
        ref_logits_256 = seg_outputs["ref_logits_256"].to(device=pe_device, dtype=pe_dtype)

        # pad ref_logits_256 to -50.0 by unpadded_hw
        B, K, H, W = ref_logits_256.shape
        up_hw = unpadded_hw.to(pe_device)
        h256 = ((up_hw[:, 0].clamp(min=0) + 3) // 4).clamp(max=H)  # (B,)
        w256 = ((up_hw[:, 1].clamp(min=0) + 3) // 4).clamp(max=W)  # (B,)
        unpad_mask = (torch.arange(H, device=pe_device)[None, None, :, None] < h256[:, None, None, None]) & \
               (torch.arange(W, device=pe_device)[None, None, None, :] < w256[:, None, None, None])

        ref_logits_256 = ref_logits_256.masked_fill(~unpad_mask, ref_logits_256.new_tensor(-50.0))  # [B, K, H, W]

        # ======== forward by segmentation modules ========
        masks_bk = ref_logits_256.reshape(B * K, 1, H, W)
        sparse_embed, dense_embed = self.prompt_encoder(
            points=None, boxes=None, masks=masks_bk,
        )  # [B*K, 1, 256, 256] -> [B*K, 256, 64, 64]

        dc_dtype, dc_device = next(self.mask_decoder.parameters()).dtype, next(self.mask_decoder.parameters()).device
        image_pe = self.prompt_encoder.get_dense_pe().to(dtype=dc_dtype, device=dc_device)
        image_embed = image_embed.to(dtype=dc_dtype, device=dc_device)
        sparse_embed = sparse_embed.to(dtype=dc_dtype, device=dc_device)
        dense_embed = dense_embed.to(dtype=dc_dtype, device=dc_device)

        low_res_masks_ls = []
        for i in range(B):
            k0, k1 = i * K, (i + 1) * K
            low_res_masks, _ = self.mask_decoder(
                image_embeddings=image_embed[i:i+1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embed[k0:k1],
                dense_prompt_embeddings=dense_embed[k0:k1],
                multimask_output=False,
            )
            low_res_masks_ls.append(low_res_masks)  # append [K, 1, 256, 256]

        low_res_masks = torch.stack(low_res_masks_ls, dim=0)  # [B, K, 1, 256, 256]
        if low_res_masks.dim() == 5 and low_res_masks.size(2) == 1:
            low_res_masks = low_res_masks.squeeze(2)  # [B, K, 256, 256]

        seg_outputs["mask_logits"] = low_res_masks
        seg_outputs["mask_sigmoid"] = torch.sigmoid(low_res_masks)

        return seg_outputs


class SlotAttnHeatmapHead(nn.Module):
    """
    Multi-slot heatmap head with per-slot confidence.
    Inputs:
      - image_embed:    (B, C=256, H=64, W=64)
      - ref_vec:        (B, K, D_llm)
      - querybook_slots:(B, K, D_qb)  [mandatory, per-slot]
    Outputs:
      - ref_logits_256: (B, K, 256, 256)   # per-slot logits (no extra channel dim)
      - conf_logits:    (B, K)             # per-slot existence logits
    """
    def __init__(
        self,
        llm_dim: int,
        querybook_dim: int = 512,
        sam_dim: int = 256,
        H: int = 64,
        W: int = 64,
        out_size: int = 256,
        num_heads: int = 4,
        refine: bool = True,
    ):
        super().__init__()
        assert sam_dim % num_heads == 0
        self.sam_dim = sam_dim
        self.H, self.W = H, W
        self.out_size = out_size
        self.num_heads = num_heads
        self.d_head = sam_dim // num_heads
        self.llm_dim = llm_dim
        self.querybook_dim = querybook_dim

        # projections
        self.q_proj = nn.Linear(llm_dim, num_heads * self.d_head, bias=False)
        self.k_proj = nn.Conv2d(sam_dim, num_heads * self.d_head, kernel_size=1, bias=False)

        # head fusion at 64x64
        if refine:
            self.fuser = nn.Sequential(
                nn.Conv2d(num_heads, num_heads, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(num_heads, 1, 3, padding=1),
            )
            nn.init.constant_(self.fuser[-1].bias, -1.8)
        else:
            self.fuser = nn.Conv2d(num_heads, 1, kernel_size=1)
            nn.init.constant_(self.fuser.bias, -1.8)

        # per-slot querybook fusion: project D_qb → D_llm, then fuse with ref_vec
        self.querybook_proj = nn.Linear(querybook_dim, llm_dim, bias=False)
        self.ref_fuse = nn.Sequential(
            nn.LayerNorm(2 * llm_dim),
            nn.Linear(2 * llm_dim, llm_dim),
            nn.SiLU(inplace=True),
        )

        # per-slot confidence head (expects llm_dim)
        self.conf_head = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, llm_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(llm_dim // 2, 1),
        )

    def forward(
        self,
        image_embed: torch.Tensor,
        ref_vec: torch.Tensor,
        querybook_slots: torch.Tensor,   # (B, K, D_qb)  <-- changed
    ):
        # shapes
        B, C, H, W = image_embed.shape
        assert C == self.sam_dim and H == self.H and W == self.W
        assert ref_vec.dim() == 3 and ref_vec.size(0) == B
        assert querybook_slots.dim() == 3 and querybook_slots.size(0) == B
        K = ref_vec.size(1)
        assert querybook_slots.size(1) == K and querybook_slots.size(2) == self.querybook_dim

        # dtype/dev alignment
        ref_vec = ref_vec.to(dtype=image_embed.dtype, device=image_embed.device)
        querybook_slots = querybook_slots.to(dtype=image_embed.dtype, device=image_embed.device)

        # fuse per-slot querybook with ref embeddings
        # (B,K,D_qb)->proj->(B,K,D_llm), then concat with ref_vec along -1, then fuse to (B,K,D_llm)
        qb = self.querybook_proj(querybook_slots)                         # (B, K, D_llm)
        fused_ref = self.ref_fuse(torch.cat([ref_vec, qb], dim=-1))       # (B, K, D_llm)

        # attention logits per head at 64x64
        Q = self.q_proj(fused_ref).view(B, K, self.num_heads, self.d_head, 1, 1)   # (B,K,Hh,Dh,1,1)
        Kmap = self.k_proj(image_embed).view(B, 1, self.num_heads, self.d_head, H, W)  # (B,1,Hh,Dh,H,W)
        logits_64_heads = (Q * Kmap).sum(dim=3) / math.sqrt(self.d_head)           # (B,K,Hh,H,W)

        # fuse heads, upsample, drop channel dim -> (B,K,OUT,OUT)
        x = logits_64_heads.view(B * K, self.num_heads, H, W)
        logits_64 = self.fuser(x)                                                  # (B*K,1,H,W)
        up = F.interpolate(logits_64, size=(self.out_size, self.out_size),
                           mode="bilinear", align_corners=False)                   # (B*K,1,OUT,OUT)
        ref_logits_256 = up.view(B, K, self.out_size, self.out_size)               # (B,K,OUT,OUT)

        # per-slot confidence from fused_ref
        conf_logits = self.conf_head(fused_ref).squeeze(-1)                        # (B,K)

        return {
            "ref_logits_256": ref_logits_256,
            "conf_logits": conf_logits,
        }


class QueryBookAttnHead(nn.Module):
    """
    Codebook project [B, L, sae_dim] -> [B, L, querybook_dim] (no bias),
    then aggregate with K learnable slot queries via a Transformer encoder.
    Returns [B, K, querybook_dim].
    """
    def __init__(
        self,
        k_slots: int = 4,
        querybook_dim: int = 512,
        activation_threshold: float = 2.0,
        sae_dim: int = 65536,
        nhead: int = 8,
        num_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.querybook_dim = int(querybook_dim)
        self.sae_dim = int(sae_dim)
        self.activation_threshold = float(activation_threshold)
        self.k_slots = int(k_slots)

        self.codebook = nn.Linear(self.sae_dim, self.querybook_dim, bias=False)

        d_model = self.querybook_dim
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ff_mult * d_model, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)

        self.slot_queries = nn.Parameter(torch.zeros(1, self.k_slots, d_model))
        nn.init.trunc_normal_(self.slot_queries, std=0.02)

    def forward(
            self,
            sae_embeds: torch.Tensor,  # [B, L, sae_dim]
            attention_mask: torch.Tensor,  # LLM-style: 1/True = valid, 0/False = pad
    ) -> torch.Tensor:  # [B, K, querybook_dim]
        B, L, _ = sae_embeds.shape
        K = self.k_slots

        x = sae_embeds
        if self.activation_threshold > 0.0:
            thr = torch.as_tensor(self.activation_threshold, dtype=x.dtype, device=x.device)
            x = x.masked_fill(x.abs() < thr, 0.0)

        # [B, L, sae_dim] -> [B, L, D]
        x = self.codebook(x)
        D = x.size(-1)

        # prepend slots
        slots = self.slot_queries.expand(B, K, D)  # [B, K, D]
        x = torch.cat([slots, x], dim=1)  # [B, K+L, D]

        # accept attention_mask in common shapes and build key_padding_mask (True = pad)
        am = attention_mask
        if am.dim() == 3 and am.size(1) == 1:  # [B,1,L] -> [B,L]
            am = am[:, 0, :]
        if am.dim() == 3 and am.size(2) == 1:  # [B,L,1] -> [B,L]
            am = am[:, :, 0]

        if am.dtype == torch.bool:
            base_kpm = ~am  # bool: True(valid) -> False, False(pad) -> True
        else:
            base_kpm = (am == 0)  # int/float: 0 -> pad(True), else -> False

        pad_slots = torch.zeros(B, K, dtype=torch.bool, device=base_kpm.device)
        key_padding_mask = torch.cat([pad_slots, base_kpm], dim=1)  # [B, K+L]

        # encoder & output slots
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, K+L, D]
        slot_repr = self.final_norm(x[:, :K, :])  # [B, K, D]
        return slot_repr


def build_VLRefSegCore(
        llm: nn.Module,
        k_slots: int,
        sae_cfg: SAEConfig,
        init_sae_ckpt: str,
        init_sam_ckpt: str,
        tokenizer,
        token_embed_dim=None,
        special_token: str = "<REF_POS>",
        dtype=torch.bfloat16,
        is_trainable=True,
):
    if token_embed_dim is None:
        token_embed_dim = llm.get_input_embeddings().embedding_dim

    query_book = QueryBookAttnHead(k_slots=k_slots)
    heatmap_head = SlotAttnHeatmapHead(llm_dim=token_embed_dim, querybook_dim=query_book.querybook_dim)

    _sam = sam_model_registry["vit_h"](checkpoint=None)
    prompt_encoder = _sam.prompt_encoder
    mask_decoder = _sam.mask_decoder
    del _sam

    sae = SAE(cfg=sae_cfg)
    if init_sae_ckpt is not None:
        sae.load_state_dict(torch.load(init_sae_ckpt, map_location="cpu")["sae"], strict=True)

    if init_sam_ckpt is not None:
        sd = torch.load(init_sam_ckpt, map_location="cpu")
        pe_sd = {k.split("prompt_encoder.", 1)[1]: v for k, v in sd.items() if k.startswith("prompt_encoder.")}
        md_sd = {k.split("mask_decoder.", 1)[1]: v for k, v in sd.items() if k.startswith("mask_decoder.")}
        prompt_encoder.load_state_dict(pe_sd, strict=False)
        mask_decoder.load_state_dict(md_sd, strict=False)

    ref_pos_id = int(tokenizer.convert_tokens_to_ids(special_token))

    sae = sae.to(dtype=dtype)
    query_book = query_book.to(dtype=dtype)
    heatmap_head = heatmap_head.to(dtype=dtype)
    prompt_encoder = prompt_encoder.to(dtype=dtype)
    mask_decoder = mask_decoder.to(dtype=dtype)

    core = VLRefSegCore(
        llm=llm,
        sae=sae,
        query_book=query_book,
        heatmap_head=heatmap_head,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        special_token_id=ref_pos_id,
        sae_hooked_layer=sae_cfg.hook_layer,
    )
    if is_trainable:
        core.train()
    else:
        core.eval()
    return core

