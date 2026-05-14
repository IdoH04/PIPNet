"""
Utility functions for MCPNet-style distribution matching on PIP-Net models.

Idea:
    D(image) = concat(Stage 3 pooled prototype scores, Stage 4 pooled prototype scores)

Then:
    class_centroid[c] = average D(image) over training images of class c

At test time:
    pred = argmin_c JS_divergence(D(image), class_centroid[c])

This file does not modify PIP-Net training. It is meant for post-hoc evaluation
of trained checkpoints first.
"""

from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader


def _unwrap(net):
    return net.module if hasattr(net, "module") else net


def normalize_distribution(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert nonnegative prototype scores into probability distributions."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = torch.clamp(x.float(), min=0.0)
    denom = x.sum(dim=1, keepdim=True)
    uniform = torch.ones_like(x) / x.shape[1]
    return torch.where(denom > eps, x / (denom + eps), uniform)


def js_divergence_batch(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Jensen-Shannon divergence between every row in p and every row in q.

    Args:
        p: [B, D]
        q: [C, D]

    Returns:
        distances: [B, C]
    """
    p = normalize_distribution(p, eps=eps)
    q = normalize_distribution(q, eps=eps)
    p_exp = p.unsqueeze(1)
    q_exp = q.unsqueeze(0)
    m = 0.5 * (p_exp + q_exp)
    kl_pm = (p_exp * (torch.log(p_exp + eps) - torch.log(m + eps))).sum(dim=-1)
    kl_qm = (q_exp * (torch.log(q_exp + eps) - torch.log(m + eps))).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


@torch.no_grad()
def extract_distribution(
    net,
    xs: torch.Tensor,
    include_stage3: bool = True,
    include_stage4: bool = True,
    normalize_parts: bool = True,
) -> torch.Tensor:
    """
    Extract a multi-level PIP-Net prototype distribution from a batch.

    Stage 4 comes from normal PIP-Net inference.
    Stage 3 comes from the multistage branch using compute_penultimate=True.
    """
    if not include_stage3 and not include_stage4:
        raise ValueError("At least one of include_stage3/include_stage4 must be True.")

    model = _unwrap(net)
    parts: List[torch.Tensor] = []

    if include_stage4:
        _, pooled_stage4, _ = net(xs, inference=True)
        pooled_stage4 = torch.clamp(pooled_stage4.float(), min=0.0)
        if normalize_parts:
            pooled_stage4 = normalize_distribution(pooled_stage4)
        parts.append(pooled_stage4)

    if include_stage3:
        if not getattr(model, "has_penultimate_branch", False):
            raise RuntimeError(
                "include_stage3=True was requested, but this model has no Stage 3 branch. "
                "Use --include_stage4 only for normal convnext_tiny_13/26 baselines."
            )
        net(xs, compute_penultimate=True)
        pooled_stage3 = getattr(model, "pooled_penultimate", None)
        if pooled_stage3 is None:
            raise RuntimeError(
                "Stage 3 pooled scores were not computed. Make sure PIPNet.forward supports "
                "compute_penultimate=True and the backbone supports return_stage='both'."
            )
        pooled_stage3 = torch.clamp(pooled_stage3.float(), min=0.0)
        if normalize_parts:
            pooled_stage3 = normalize_distribution(pooled_stage3)
        parts.append(pooled_stage3)

    dist = torch.cat(parts, dim=1)
    return normalize_distribution(dist)


@torch.no_grad()
def compute_class_centroids(
    net,
    loader: DataLoader,
    num_classes: int,
    device,
    include_stage3: bool = True,
    include_stage4: bool = True,
    normalize_parts: bool = True,
) -> torch.Tensor:
    """Compute class-average prototype distributions."""
    net.eval()
    sums: Optional[torch.Tensor] = None
    counts = torch.zeros(num_classes, device=device, dtype=torch.float32)

    for batch in loader:
        if len(batch) == 3:
            xs, _, ys = batch
        else:
            xs, ys = batch
        xs = xs.to(device)
        ys = ys.to(device)
        dists = extract_distribution(
            net,
            xs,
            include_stage3=include_stage3,
            include_stage4=include_stage4,
            normalize_parts=normalize_parts,
        )
        if sums is None:
            sums = torch.zeros(num_classes, dists.shape[1], device=device, dtype=torch.float32)
        for c in range(num_classes):
            mask = ys == c
            if torch.any(mask):
                sums[c] += dists[mask].sum(dim=0)
                counts[c] += mask.sum().float()

    if sums is None:
        raise RuntimeError("No batches were found while computing centroids.")

    centroids = sums / counts.clamp_min(1.0).unsqueeze(1)
    missing = counts == 0
    if torch.any(missing):
        centroids[missing] = torch.ones_like(centroids[missing]) / centroids.shape[1]
    return normalize_distribution(centroids)


@torch.no_grad()
def predict_distribution_matching(
    net,
    xs: torch.Tensor,
    centroids: torch.Tensor,
    include_stage3: bool = True,
    include_stage4: bool = True,
    normalize_parts: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict labels using nearest class centroid by JS divergence."""
    dists = extract_distribution(
        net,
        xs,
        include_stage3=include_stage3,
        include_stage4=include_stage4,
        normalize_parts=normalize_parts,
    )
    distances = js_divergence_batch(dists, centroids.to(xs.device))
    preds = torch.argmin(distances, dim=1)
    return preds, distances


@torch.no_grad()
def eval_distribution_matching(
    net,
    loader: DataLoader,
    centroids: torch.Tensor,
    device,
    include_stage3: bool = True,
    include_stage4: bool = True,
    normalize_parts: bool = True,
) -> Dict:
    """Evaluate distribution matching on a dataloader."""
    net.eval()
    all_preds: List[int] = []
    all_true: List[int] = []
    all_min_dist: List[float] = []

    for batch in loader:
        if len(batch) == 3:
            xs, _, ys = batch
        else:
            xs, ys = batch
        xs = xs.to(device)
        ys = ys.to(device)
        preds, distances = predict_distribution_matching(
            net,
            xs,
            centroids,
            include_stage3=include_stage3,
            include_stage4=include_stage4,
            normalize_parts=normalize_parts,
        )
        min_dist = torch.gather(distances, dim=1, index=preds.unsqueeze(1)).squeeze(1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_true.extend(ys.detach().cpu().tolist())
        all_min_dist.extend(min_dist.detach().cpu().tolist())

    correct = sum(int(p == y) for p, y in zip(all_preds, all_true))
    total = len(all_true)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "preds": all_preds,
        "true": all_true,
        "min_js_distance": all_min_dist,
    }


def save_centroids(path: str, centroids: torch.Tensor, meta: Optional[Dict] = None) -> None:
    torch.save({"centroids": centroids.detach().cpu(), "meta": meta or {}}, path)


def load_centroids(path: str, device=None) -> Tuple[torch.Tensor, Dict]:
    ckpt = torch.load(path, map_location=device)
    centroids = ckpt["centroids"].to(device) if device is not None else ckpt["centroids"]
    return centroids, ckpt.get("meta", {})


# -----------------------------------------------------------------------------
# Differentiable helpers for Version A: Stage3+Stage4 distribution training loss
# -----------------------------------------------------------------------------

def extract_distribution_from_pooled(
    pooled_stage3: Optional[torch.Tensor],
    pooled_stage4: Optional[torch.Tensor],
    include_stage3: bool = True,
    include_stage4: bool = True,
    normalize_parts: bool = True,
) -> torch.Tensor:
    """Build a differentiable distribution from already-computed pooled scores.

    This is used inside the training loop. Unlike extract_distribution(), this
    function does not call the model and does not use torch.no_grad(). Gradients
    can flow through the pooled scores into the prototype/backbone path.
    """
    if not include_stage3 and not include_stage4:
        raise ValueError("At least one of include_stage3/include_stage4 must be True.")

    parts: List[torch.Tensor] = []

    if include_stage3:
        if pooled_stage3 is None:
            raise RuntimeError("include_stage3=True but pooled_stage3 is None.")
        s3 = torch.clamp(pooled_stage3.float(), min=0.0)
        if normalize_parts:
            s3 = normalize_distribution(s3)
        parts.append(s3)

    if include_stage4:
        if pooled_stage4 is None:
            raise RuntimeError("include_stage4=True but pooled_stage4 is None.")
        s4 = torch.clamp(pooled_stage4.float(), min=0.0)
        if normalize_parts:
            s4 = normalize_distribution(s4)
        parts.append(s4)

    return normalize_distribution(torch.cat(parts, dim=1))


def class_aware_distribution_loss(
    dists: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    margin: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """MCPNet/CCD-style class-aware distribution loss.

    Positive term: pull each image distribution toward its true class centroid.
    Negative term: push it away from other class centroids up to a margin.

    Args:
        dists: [B, D] image distributions.
        labels: [B] integer labels.
        centroids: [num_classes, D] class centroid distributions.
        margin: hinge margin for negative classes.
    """
    if dists.ndim != 2:
        raise ValueError(f"dists must be [B, D], got {tuple(dists.shape)}")
    if centroids.ndim != 2:
        raise ValueError(f"centroids must be [C, D], got {tuple(centroids.shape)}")
    if dists.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"Distribution dim mismatch: dists has {dists.shape[1]}, centroids has {centroids.shape[1]}"
        )

    labels = labels.long()
    centroids = centroids.to(dists.device)
    distances = js_divergence_batch(dists, centroids, eps=eps)  # [B, C]

    true_dist = distances.gather(1, labels.view(-1, 1)).squeeze(1)
    positive_loss = true_dist.mean()

    num_classes = centroids.shape[0]
    class_ids = torch.arange(num_classes, device=dists.device).view(1, -1)
    wrong_mask = class_ids != labels.view(-1, 1)
    negative_hinge = torch.relu(float(margin) - distances)
    negative_loss = negative_hinge[wrong_mask].mean() if torch.any(wrong_mask) else torch.zeros_like(positive_loss)

    return positive_loss + negative_loss


def distribution_loss_from_pooled(
    pooled_stage3: Optional[torch.Tensor],
    pooled_stage4: Optional[torch.Tensor],
    labels: torch.Tensor,
    centroids: torch.Tensor,
    include_stage3: bool = True,
    include_stage4: bool = True,
    normalize_parts: bool = True,
    margin: float = 0.05,
) -> torch.Tensor:
    """Convenience wrapper used by pipnet/train.py."""
    dists = extract_distribution_from_pooled(
        pooled_stage3=pooled_stage3,
        pooled_stage4=pooled_stage4,
        include_stage3=include_stage3,
        include_stage4=include_stage4,
        normalize_parts=normalize_parts,
    )
    return class_aware_distribution_loss(dists, labels, centroids, margin=margin)
