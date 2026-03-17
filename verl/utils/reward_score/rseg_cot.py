import re
import torch


def format_reward(tokens: str) -> float:
    """Compute a format score in {0.0, 0.9, 1.0}:
    - Return 0.0 unless ALL hard rules pass:
      (1) exactly one <think>...</think> with non-empty content,
      (2) exactly one <REF_POS>,
      (3) <REF_POS> appears after </think>.
    - If hard rules pass, start from 1.0 and downgrade to 0.9 if either:
      (a) the <think> content is overly long (> 2048 chars), or
      (b) there is any non-whitespace text before <think> or after <REF_POS>.
    """
    think_blocks = list(re.finditer(r"<think>(.*?)</think>", tokens, flags=re.DOTALL))
    if len(think_blocks) != 1:
        return 0.0
    think_match = think_blocks[0]
    think_content = think_match.group(1)
    if not re.search(r"\S", think_content or ""):
        return 0.0

    ref_tag = "<REF_POS>"
    if tokens.count(ref_tag) != 1:
        return 0.0
    ref_idx = tokens.find(ref_tag)
    if ref_idx < think_match.end():
        return 0.0

    long_think = len(think_content) > 2048
    has_non_ws_before = bool(re.search(r"\S", tokens[:think_match.start()]))
    between = tokens[think_match.end():ref_idx]
    has_long_between = len(between.split()) > 10

    return 0.9 if (long_think or has_non_ws_before or has_long_between) else 1.0


def mask_reward(masks: torch.Tensor, gt_masks: torch.Tensor) -> float:
    """
    Compute a robust segmentation reward by combining soft IoU, soft Dice, and hard IoU.
    Inputs are (B, 1, 256, 256): `masks` are sigmoid probabilities; `gt_masks` are [0,1] floats.
    Returns a scalar Python float in [0, 1].
    """
    eps = 1e-6
    thr = 0.5

    # Unify dtype & device; clamp for safety.
    masks = masks.clamp(0.0, 1.0)
    gt_masks = gt_masks.contiguous().to(device=masks.device, dtype=masks.dtype).clamp(0.0, 1.0)

    reduce_dims = (1, 2, 3)

    # Soft IoU
    inter_soft = (masks * gt_masks).sum(dim=reduce_dims)
    union_soft = (masks + gt_masks - masks * gt_masks).sum(dim=reduce_dims) + eps
    soft_iou = inter_soft / union_soft  # [B]

    # Soft Dice
    denom_dice = (masks + gt_masks).sum(dim=reduce_dims) + eps
    soft_dice = (2.0 * inter_soft) / denom_dice  # [B]

    # Hard IoU (thresholded)
    pred_bin = (masks >= thr)
    gt_bin = (gt_masks >= 0.5)
    inter_hard = (pred_bin & gt_bin).sum(dim=reduce_dims).to(gt_masks.dtype)
    union_hard = (pred_bin | gt_bin).sum(dim=reduce_dims).to(gt_masks.dtype) + eps
    hard_iou = inter_hard / union_hard  # [B]

    # Weighted fusion
    reward = 0.5 * soft_iou + 0.2 * soft_dice + 0.3 * hard_iou  # [B]
    return reward.mean().item()


def rseg_cot_compute_score(tokens: str, masks: torch.Tensor, gt_masks: torch.Tensor) -> float:
    format_score = format_reward(tokens)
    mask_score = mask_reward(masks, gt_masks)
    return 0.3 * format_score + 0.7 * mask_score
