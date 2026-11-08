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
    return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}" if len(values) > 1 else f"{values.mean():.4f} ± nan"


def paired_delta(df, left, right, metric):
    a = df[df.model == left][["seed", metric]].rename(columns={metric: "left"})
    b = df[df.model == right][["seed", metric]].rename(columns={metric: "right"})
    merged = a.merge(b, on="seed", how="inner")
    if merged.empty:
        return float("nan"), float("nan")
    delta = merged["right"] - merged["left"]
    return float(delta.mean()), float(delta.std(ddof=1)) if len(delta) > 1 else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/corrected_v1")
    args = parser.parse_args()
    root = Path(args.results_dir)
    metrics_path = root / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    df = pd.read_csv(metrics_path)

    print("\nQ1 — Domain shift: ID validation so với OOD validation")
    for model in sorted(df.model.unique()):
        subset = df[df.model == model]
        d_acc = subset["id_accuracy"] - subset["ood_accuracy"]
        d_auc = subset["id_auroc"] - subset["ood_auroc"]
        d_loss = subset["ood_loss"] - subset["id_loss"]
        rel_acc = d_acc / subset["id_accuracy"].replace(0, np.nan) * 100
        print(
            f"{model:16s} ID acc={mean_std(subset.id_accuracy)} "
            f"OOD acc={mean_std(subset.ood_accuracy)} "
            f"DeltaAcc={mean_std(d_acc)} "
            f"relative Acc drop={mean_std(rel_acc)}% "
            f"DeltaAUROC={mean_std(d_auc)} DeltaLoss(OOD-ID)={mean_std(d_loss)}"
        )

    print("\nQ2 — Paired augmentation deltas (augmentation - baseline)")
    for baseline, augmented in [("erm", "erm_aug"), ("group_dro", "group_dro_aug")]:
        for metric in ["ood_accuracy", "id_accuracy", "ood_loss", "id_loss"]:
            delta, std = paired_delta(df, baseline, augmented, metric)
            print(f"{baseline} -> {augmented}: {metric}: delta={delta:+.4f}, std={std:.4f}")

    print("\nQ3 — Hospital risks and accuracies (mean ± std across seeds)")
    for model in sorted(df.model.unique()):
        subset = df[df.model == model]
        print(
            f"{model:16s} avg={mean_std(subset.avg_risk)} "
            f"worst={mean_std(subset.worst_risk)} "
            f"worst_seen={mean_std(subset.worst_risk_seen)} "
            f"worst_unseen={mean_std(subset.worst_risk_unseen)}"
        )
        for h in range(5):
            print(
                f"  hospital {h}: acc={mean_std(subset[f'acc_h{h}'])}, "
                f"risk={mean_std(subset[f'risk_h{h}'])}"
            )

    # Plot 1: ID vs OOD validation metrics
    summary = df.groupby("model")[["id_accuracy", "ood_accuracy"]].mean()
    ax = summary.plot(kind="bar", figsize=(9, 5), rot=20)
    ax.set_ylabel("Accuracy")
    ax.set_title("ID Validation vs OOD Validation Accuracy")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(root / "q1_id_ood_accuracy.png", dpi=160)
    plt.close()

    # Plot 2: accuracy per hospital
    acc_cols = [f"acc_h{h}" for h in range(5)]
    ax = df.groupby("model")[acc_cols].mean().plot(kind="bar", figsize=(10, 5), rot=20)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Hospital")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(root / "q3_hospital_accuracy.png", dpi=160)
    plt.close()

    # Plot 3: average, worst seen and worst unseen risks
    risk_cols = ["avg_risk", "worst_risk_seen", "worst_risk_unseen"]
    ax = df.groupby("model")[risk_cols].mean().plot(kind="bar", figsize=(10, 5), rot=20)
    ax.set_ylabel("Cross-Entropy Risk")
    ax.set_title("Average Risk vs Worst Seen / Unseen Risk")
    plt.tight_layout()
    plt.savefig(root / "q3_risk_comparison.png", dpi=160)
    plt.close()

    # Plot 4: ID and OOD validation loss histories
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
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(root / "q1_loss_curves.png", dpi=160)
    plt.close()

    print(f"\nAnalysis outputs saved in {root.resolve()}")


if __name__ == "__main__":
    main()
