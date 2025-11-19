"""
Cosine-only retrieval evaluation utilities (single/dual-backbone).
Outputs: Recall@1, Recall@5, mAP, mAP@1, mAP@5.
- Handles models that return either embeddings OR (embeddings, logits).
"""

# from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize
from typing import Tuple, Dict, Any

# -----------------------------
# Metrics (ranking-based)
# -----------------------------

def compute_recall_at_k(sim_matrix: np.ndarray, q_labels: np.ndarray, g_labels: np.ndarray, k: int) -> float:
    hit = 0
    k = int(k)
    for i in range(len(q_labels)):
        order = np.argsort(-sim_matrix[i])[:k]
        if (g_labels[order] == q_labels[i]).any():
            hit += 1
    return hit / len(q_labels)


def average_precision_at_k(sim_row: np.ndarray, q_label: int, g_labels: np.ndarray, k: int | None = None) -> float:
    order = np.argsort(-sim_row)
    rel_full = (g_labels[order] == q_label).astype(np.int32)
    pos_total = int(rel_full.sum())
    if pos_total == 0:
        return np.nan

    cutoff = len(order) if k is None else int(k)
    rel = rel_full[:cutoff]
    cum_rel = np.cumsum(rel)
    ranks = np.arange(1, len(rel) + 1)
    prec = cum_rel / ranks

    denom = min(pos_total, cutoff)
    ap = float((prec * rel).sum() / max(denom, 1))
    return ap


def compute_mean_ap(sim_matrix: np.ndarray, q_labels: np.ndarray, g_labels: np.ndarray, k: int | None = None) -> float:
    aps: list[float] = []
    for i in range(len(q_labels)):
        ap = average_precision_at_k(sim_matrix[i], int(q_labels[i]), g_labels, k)
        if not np.isnan(ap):
            aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


# -----------------------------
# Core evaluators (single model)
# -----------------------------
@torch.no_grad()
def _extract_embeddings(model: torch.nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Patched: robustly handles models that return either `emb` or `(emb, logits)`."""
    model.eval(); model.to(device)
    feats, labs = [], []
    for images, labels in tqdm(loader, desc="Extract"):
        images = images.to(device, non_blocking=True)
        out = model(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
        emb = out.detach().cpu().numpy()
        feats.append(emb)
        labs.append(labels.cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    labs = np.concatenate(labs, axis=0)
    return feats, labs


@torch.no_grad()
def _cosine_similarity(query_feats: np.ndarray, gallery_feats: np.ndarray) -> np.ndarray:
    q = normalize(query_feats, axis=1)
    g = normalize(gallery_feats, axis=1)
    return np.matmul(q, g.T)


def compute_classwise_metrics(sim_matrix, q_labels, g_labels, idx_to_class: Dict[int, str], ks=[1, 5]):
    classes = np.unique(q_labels)
    results: Dict[str, Dict[str, float]] = {}

    for cls in classes:
        mask = (q_labels == cls)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue

        sub_sim = sim_matrix[idxs]
        sub_q   = q_labels[idxs]

        cls_map = compute_mean_ap(sub_sim, sub_q, g_labels)
        cls_recalls = {f"Recall@{k}": compute_recall_at_k(sub_sim, sub_q, g_labels, k) for k in ks}

        name = idx_to_class.get(int(cls), f"class_{int(cls)}")
        results[name] = {"mAP": cls_map, **cls_recalls}

    mean_map = np.mean([v["mAP"] for v in results.values()]) if results else 0.0
    mean_recalls = {f"Recall@{k}": (np.mean([v[f"Recall@{k}"] for v in results.values()]) if results else 0.0) for k in ks}
    return results, {"mAP": mean_map, **mean_recalls}


@torch.no_grad()
def evaluate_retrieval(model, query_loader, gallery_loader, device, idx_to_class=None):
    """Cosine-only metrics: Recall@1/5, mAP, mAP@1/5. Also prints class-wise if idx_to_class is provided."""
    q_feats, q_labels = _extract_embeddings(model, query_loader, device)
    g_feats, g_labels = _extract_embeddings(model, gallery_loader, device)

    sim = _cosine_similarity(q_feats, g_feats)

    r1  = compute_recall_at_k(sim, q_labels, g_labels, 1)
    r5  = compute_recall_at_k(sim, q_labels, g_labels, 5)
    mAP  = compute_mean_ap(sim, q_labels, g_labels, k=None)

    print("\n[Cosine] Retrieval metrics")
    print(f"  Recall@1 : {r1*100:.2f}%")
    print(f"  Recall@5 : {r5*100:.2f}%")
    print(f"  mAP      : {mAP*100:.2f}%")


    if idx_to_class is not None:
        per_cls, per_cls_mean = compute_classwise_metrics(sim, q_labels, g_labels, idx_to_class, ks=[1, 5])
        print("\n==== Class-wise Metrics ====")
        for name, vals in per_cls.items():
            print(f"{name:20s} | mAP={vals['mAP']:.4f} | Recall@1={vals['Recall@1']:.4f} | Recall@5={vals['Recall@5']:.4f}")
        print("\n==== Mean (across classes) ====")
        print(f"mAP={per_cls_mean['mAP']:.4f} | Recall@1={per_cls_mean['Recall@1']:.4f} | Recall@5={per_cls_mean['Recall@5']:.4f}")

    return r1, r5, mAP
