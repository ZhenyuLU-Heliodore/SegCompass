import torch

from verl.utils.reward_score.rseg_cot import format_reward, mask_reward
from verl.utils.torch_functional import pairwise_soft_iou
from scipy.optimize import linear_sum_assignment


def scots_compute_score(
    tokens: str,
    masks: torch.Tensor,
    gt_masks: torch.Tensor,
    n_multi_objects: torch.Tensor,
    conf_logits: torch.Tensor,
):
    """
    Compute scalar score = 0.3*format_reward(tokens) + 0.7*mask_score.

    Args:
        tokens: prompt/response text.
        masks: predicted masks, shape [K, H, W].
        gt_masks: ground-truth masks, shape [G, H, W].
        n_multi_objects: scalar tensor; -1 for single-object, else number of GT used (G).
        conf_logits: per-prediction confidence logits, shape [K].

    Returns:
        float: final score.
    """
    format_score = format_reward(tokens)

    n_gt = int(n_multi_objects.item())
    if n_gt == -1:
        mask_score = float(mask_reward(masks.unsqueeze(0), gt_masks.unsqueeze(0)))
    else:
        keep = (conf_logits >= 0.5).nonzero(as_tuple=False).squeeze(-1)
        pred = masks[keep] if keep.numel() > 0 else masks.new_zeros((0, masks.size(-2), masks.size(-1)))
        gt = gt_masks[:n_gt]
        mask_score = _hungarian_mean_iou_single(pred, gt, n_gt)

    return 0.3 * float(format_score) + 0.7 * float(mask_score)


def _hungarian_mean_iou_single(pred_khw: torch.Tensor, gt_khw: torch.Tensor, n_gt: int) -> float:
    gt_khw = gt_khw[:n_gt]
    K = pred_khw.shape[0]
    G = gt_khw.shape[0]

    if K == 0 and G == 0:
        return 1.0
    if K == 0 or G == 0:
        return 0.0

    iou = pairwise_soft_iou(pred_khw, gt_khw)  # [K, G]
    M = max(K, G)
    iou_pad = iou.new_zeros((M, M))
    iou_pad[:K, :G] = iou

    r, c = linear_sum_assignment((1.0 - iou_pad).detach().cpu().numpy())
    return float(iou_pad[r, c].mean().item())


if __name__ == "__main__":
    import torch

    # Case 1: K=G=2，完美匹配 → mask_score=1.0 → score=0.7
    H = W = 4
    gt1 = torch.zeros(2, H, W)
    gt1[0, 0:2, 0:2] = 1.0
    gt1[1, 2:4, 2:4] = 1.0

    pred1 = gt1.clone()
    conf1 = torch.tensor([1.0, 2.0])     # keep both
    n_gt1 = torch.tensor(2)

    s1 = scots_compute_score("", pred1, gt1, n_gt1, conf1)
    print(f"Case 1 score = {s1:.6f}  expected ≈ 0.700000 (mask≈1.0)")

    # Case 2: 过预测 K=3, G=1；只有一个预测能对上，其余两个被零填充罚分
    # IoU_pad 匹配平均 = (1 + 0 + 0)/3 = 1/3 → score=0.7*(1/3)=0.233333
    gt2 = torch.zeros(1, H, W)
    gt2[0, 0:2, 0:2] = 1.0

    pred2 = torch.zeros(3, H, W)
    pred2[0, 0:2, 0:2] = 1.0      # perfect match
    # pred2[1], pred2[2] are zeros -> IoU 0
    conf2 = torch.tensor([1.0, 1.0, 1.0])  # keep all
    n_gt2 = torch.tensor(1)

    s2 = scots_compute_score("", pred2, gt2, n_gt2, conf2)
    print(f"Case 2 score = {s2:.6f}  expected ≈ 0.233333 (mask≈0.333333)")

    # Case 3: 置信度筛掉一部分后 K==G==2，构造 IoU 分别为 0.6 和 0.2 → 平均 0.4 → score=0.28
    gt3 = torch.zeros(2, H, W)
    gt3[0, 0:2, 0:2] = 1.0  # area=4
    gt3[1, 3, 3] = 1.0      # area=1

    pred3 = torch.zeros(3, H, W)
    pred3[0, 0:2, 0:2] = 0.6  # IoU vs gt3[0] = 0.6
    pred3[1, 3, 3] = 0.2      # IoU vs gt3[1] = 0.2
    pred3[2, :, :] = 0.0      # will be filtered out by conf
    conf3 = torch.tensor([0.9, 0.8, 0.1])  # keep idx 0,1; drop idx 2
    n_gt3 = torch.tensor(2)

    s3 = scots_compute_score("", pred3, gt3, n_gt3, conf3)
    print(f"Case 3 score = {s3:.6f}  expected ≈ 0.280000 (mask≈0.4)")

    # Case 4: 全部置信度<0.5 → 过滤后 K=0, G>0 → mask_score=0.0 → score=0.0
    gt4 = gt2.clone()  # G=1
    pred4 = torch.zeros(2, H, W)  # whatever
    conf4 = torch.tensor([-1.0, -0.1])  # drop all
    n_gt4 = torch.tensor(1)

    s4 = scots_compute_score("", pred4, gt4, n_gt4, conf4)
    print(f"Case 4 score = {s4:.6f}  expected ≈ 0.000000 (mask≈0.0)")

    # Case 5: 过滤后 K=0 且 n_gt=0（空对空）→ mask_score=1.0 → score=0.7
    gt5 = torch.zeros(0, H, W)   # G=0
    pred5 = torch.zeros(2, H, W) # will be dropped by conf
    conf5 = torch.tensor([-2.0, -3.0])  # drop all → K'=0
    n_gt5 = torch.tensor(0)

    s5 = scots_compute_score("", pred5, gt5, n_gt5, conf5)
    print(f"Case 5 score = {s5:.6f}  expected ≈ 0.700000 (mask≈1.0)")