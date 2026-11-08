"""Đánh giá classification metrics và rủi ro theo hospital."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, dataloader, device, n_groups, hospital_idx):
    model.eval()
    preds_all, probs_all, labels_all, groups_all, losses_all = [], [], [], [], []

    for x, y, metadata in tqdm(dataloader, desc="Evaluating", leave=False):
        x = x.to(device)
        y = y.to(device).long()
        groups = metadata[:, hospital_idx].long().cpu().numpy()

        logits = model(x)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        losses = F.cross_entropy(logits, y, reduction="none")

        preds_all.append(preds.cpu().numpy())
        probs_all.append(probs.cpu().numpy())
        labels_all.append(y.cpu().numpy())
        groups_all.append(groups)
        losses_all.append(losses.cpu().numpy())

    if not labels_all:
        return {
            "accuracy": float("nan"), "auroc": float("nan"), "loss": float("nan"),
            "avg_group_risk": float("nan"), "worst_group_risk": float("nan"),
            "per_group_risk": {}, "per_group_acc": {}, "n_samples": 0,
        }

    preds = np.concatenate(preds_all)
    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    groups = np.concatenate(groups_all)
    losses = np.concatenate(losses_all)

    try:
        auroc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan")
    except ValueError:
        auroc = float("nan")

    per_group_risk, per_group_acc = {}, {}
    for group_id in range(n_groups):
        mask = groups == group_id
        if mask.any():
            per_group_risk[int(group_id)] = float(losses[mask].mean())
            per_group_acc[int(group_id)] = float(accuracy_score(labels[mask], preds[mask]))

    risks = list(per_group_risk.values())
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "auroc": auroc,
        "loss": float(losses.mean()),
        "avg_group_risk": float(np.mean(risks)) if risks else float("nan"),
        "worst_group_risk": float(np.max(risks)) if risks else float("nan"),
        "per_group_risk": per_group_risk,
        "per_group_acc": per_group_acc,
        "n_samples": int(len(labels)),
    }


@torch.no_grad()
def evaluate_all_hospitals(model, splits, device, n_groups, hospital_idx, seen):
    """Gộp hospital 0/3/4 từ id_val, 1 từ val và 2 từ test."""
    parts = {
        split_name: evaluate(model, splits[split_name], device, n_groups, hospital_idx)
        for split_name in ("id_val", "val", "test")
    }

    risk, acc = {}, {}
    for split_name, metrics in parts.items():
        for hospital, value in metrics["per_group_risk"].items():
            if hospital in risk:
                raise ValueError(f"Hospital {hospital} xuất hiện ở nhiều split; không thể gộp.")
            risk[int(hospital)] = float(value)
        for hospital, value in metrics["per_group_acc"].items():
            acc[int(hospital)] = float(value)

    expected = set(range(n_groups))
    found = set(risk)
    if found != expected:
        raise ValueError(
            f"Không thu thập đủ hospital. Mong đợi {sorted(expected)}, nhận được {sorted(found)}."
        )

    seen = {int(h) for h in seen}
    unseen = expected - seen
    if not seen or not unseen:
        raise ValueError("Cần có cả nhóm seen và unseen để tính Q3.")

    return {
        "parts": parts,
        "risk": risk,
        "acc": acc,
        "avg_risk": float(np.mean(list(risk.values()))),
        "worst_risk": float(max(risk.values())),
        "worst_risk_seen": float(max(risk[h] for h in seen)),
        "worst_risk_unseen": float(max(risk[h] for h in unseen)),
        "worst_acc": float(min(acc.values())),
    }
