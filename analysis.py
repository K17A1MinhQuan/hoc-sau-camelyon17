"""Phân tích Q1-Q3 từ metrics.csv và *_history.json; không huấn luyện lại."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mean_std(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return "nan"
    if len(values) == 1:
        return f"{values.mean():.4f} ± n/a"
    return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"


def paired_delta(df, baseline, augmented, metric):
    a = df[df.model == baseline][["seed", metric]].rename(columns={metric: "baseline"})
    b = df[df.model == augmented][["seed", metric]].rename(columns={metric: "augmented"})
    merged = a.merge(b, on="seed", how="inner")
    if merged.empty:
        return float("nan"), float("nan"), 0
    delta = merged["augmented"] - merged["baseline"]
    return float(delta.mean()), float(delta.std(ddof=1)) if len(delta) > 1 else float("nan"), len(delta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/corrected_v1")
    args = parser.parse_args()
    root = Path(args.results_dir)
    metrics_path = root / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    df = pd.read_csv(metrics_path)
    if df.empty:
        raise ValueError(f"{metrics_path} không có dòng kết quả.")

    print("\nQ1 — Domain shift: ID validation so với OOD validation (Hospital 1)")
    for model in sorted(df.model.unique()):
        subset = df[df.model == model]
        d_acc = subset["id_accuracy"] - subset["ood_accuracy"]
        d_auc = subset["id_auroc"] - subset["ood_auroc"]
        d_loss = subset["ood_loss"] - subset["id_loss"]
        rel_acc = d_acc / subset["id_accuracy"].replace(0, np.nan) * 100
        print(
            f"{model:16s} ID acc={mean_std(subset.id_accuracy)} "
            f"OOD acc={mean_std(subset.ood_accuracy)} "
            f"ID-OOD gap={mean_std(d_acc)} "
            f"relative accuracy drop={mean_std(rel_acc)}% "
            f"ID-OOD AUROC gap={mean_std(d_auc)} "
            f"OOD-ID loss change={mean_std(d_loss)}"
        )

    print("\nQ2 — Paired augmentation deltas (augmentation - baseline)")
    for baseline, augmented in [("erm", "erm_aug"), ("group_dro", "group_dro_aug")]:
        for metric in ["ood_accuracy", "ood_auroc", "ood_loss", "id_accuracy", "id_loss"]:
            delta, std, n = paired_delta(df, baseline, augmented, metric)
            print(f"{baseline} -> {augmented}: {metric}: delta={delta:+.4f}, std={std:.4f}, paired_seeds={n}")

    print("\nQ3 — Average Risk và Worst-Domain Risk trên validation domains")
    if "avg_risk_val4" in df.columns:
        risk_cols = ["avg_risk_val4", "worst_risk_val4",
                     "worst_risk_seen_val4", "worst_risk_unseen_val4"]
        for model in sorted(df.model.unique()):
            subset = df[df.model == model]
            print(f"{model:16s} " + " | ".join(f"{c}={mean_std(subset[c])}" for c in risk_cols))
        print("Ghi chú: Q3 validation chỉ gồm H0,H3,H4 (ID-val) và H1 (OOD-val), không dùng test H2.")
    else:
        print("CẢNH BÁO: metrics.csv cũ không có avg_risk_val4/worst_risk_val4.")
        print("Các cột avg_risk/worst_risk cũ gộp cả test H2, chỉ được xem là hậu nghiệm, không dùng làm Q3 chính.")
        for model in sorted(df.model.unique()):
            subset = df[df.model == model]
            print(f"{model:16s} avg_all5={mean_std(subset.avg_risk)} worst_all5={mean_std(subset.worst_risk)}")

    # Q1
    summary = df.groupby("model")[["id_accuracy", "ood_accuracy"]].mean()
    ax = summary.plot(kind="bar", figsize=(9, 5), rot=20)
    ax.set_ylabel("Accuracy")
    ax.set_title("ID Validation vs OOD Validation Accuracy")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(root / "q1_id_ood_accuracy.png", dpi=160)
    plt.close()

    # Q3 primary chart: validation-only risk.
    if "avg_risk_val4" in df.columns:
        cols = ["avg_risk_val4", "worst_risk_val4"]
        ax = df.groupby("model")[cols].mean().plot(kind="bar", figsize=(9, 5), rot=20)
        ax.set_ylabel("Cross-Entropy Risk")
        ax.set_title("Average Risk vs Worst-Domain Risk (Validation Hospitals)")
        plt.tight_layout()
        plt.savefig(root / "q3_validation_risk.png", dpi=160)
        plt.close()

        val_acc_cols = [f"acc_val_h{h}" for h in [0, 1, 3, 4] if f"acc_val_h{h}" in df.columns]
        if val_acc_cols:
            ax = df.groupby("model")[val_acc_cols].mean().plot(kind="bar", figsize=(10, 5), rot=20)
            ax.set_ylabel("Accuracy")
            ax.set_title("Validation Accuracy by Hospital (H0, H1, H3, H4)")
            ax.set_ylim(0, 1)
            plt.tight_layout()
            plt.savefig(root / "q3_validation_hospital_accuracy.png", dpi=160)
            plt.close()

    # Post-hoc five-hospital chart is clearly labeled and never used for selection.
    acc_cols = [f"acc_h{h}" for h in range(5) if f"acc_h{h}" in df.columns]
    if acc_cols:
        ax = df.groupby("model")[acc_cols].mean().plot(kind="bar", figsize=(10, 5), rot=20)
        ax.set_ylabel("Accuracy")
        ax.set_title("Post-hoc Accuracy by Hospital (includes held-out test H2)")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(root / "q3_all5_posthoc_accuracy.png", dpi=160)
        plt.close()

    # Loss histories
    fig, ax = plt.subplots(figsize=(10, 6))
    for history_file in sorted(root.glob("*_history.json")):
        try:
            history = json.loads(history_file.read_text(encoding="utf-8"))
            train_loss = history.get("train_loss", [])
            id_loss = [x.get("loss", np.nan) for x in history.get("id_val", [])]
            ood_loss = [x.get("loss", np.nan) for x in history.get("ood_val", [])]
            n = min(len(train_loss), len(id_loss), len(ood_loss))
            if n:
                epochs = np.arange(1, n + 1)
                ax.plot(epochs, train_loss[:n], label=f"{history_file.stem} train")
                ax.plot(epochs, id_loss[:n], "--", label=f"{history_file.stem} ID val")
                ax.plot(epochs, ood_loss[:n], ":", label=f"{history_file.stem} OOD val")
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training / ID Validation / OOD Validation Loss")
    if ax.lines:
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(root / "q1_loss_curves.png", dpi=160)
    plt.close()

    print(f"\nAnalysis outputs saved in {root.resolve()}")


if __name__ == "__main__":
    main()
