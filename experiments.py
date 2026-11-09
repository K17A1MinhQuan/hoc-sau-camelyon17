"""Run Camelyon17 experiments with resumable checkpoints and leakage-safe evaluation."""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset import get_dataloaders, get_domain_stats
from evaluate import evaluate, evaluate_all_hospitals, evaluate_validation_domains
from model import get_model
from train import train_one_epoch_erm, train_one_epoch_group_dro

EXPERIMENT_CONFIGS = {
    "erm": {"method": "erm", "augmentation": False},
    "erm_aug": {"method": "erm", "augmentation": True},
    "group_dro": {"method": "group_dro", "augmentation": False},
    "group_dro_aug": {"method": "group_dro", "augmentation": True},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def atomic_torch_save(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temp_path = str(path) + ".tmp"
    try:
        torch.save(obj, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (float, int)):
        return obj
    return obj


def flatten_metrics(run_name, seed, parts, val4, all5):
    idv, oodv, test = parts["id_val"], parts["val"], parts["test"]
    row = {
        "model": run_name, "seed": seed,
        "id_accuracy": idv["accuracy"], "id_auroc": idv["auroc"], "id_loss": idv["loss"],
        "ood_accuracy": oodv["accuracy"], "ood_auroc": oodv["auroc"], "ood_loss": oodv["loss"],
        "test_accuracy": test["accuracy"], "test_auroc": test["auroc"], "test_loss": test["loss"],
        # Q3 chính: chỉ dùng validation domains H0, H3, H4, H1; không có test H2.
        "avg_risk_val4": val4["avg_risk"],
        "worst_risk_val4": val4["worst_risk"],
        "worst_risk_seen_val4": val4["worst_risk_seen"],
        "worst_risk_unseen_val4": val4["worst_risk_unseen"],
        "worst_acc_val4": val4["worst_acc"],
        # Báo cáo 5 hospital hậu nghiệm; không dùng để chọn mô hình/hyperparameter.
        "avg_risk_all5_posthoc": all5["avg_risk"],
        "worst_risk_all5_posthoc": all5["worst_risk"],
        "worst_risk_seen_all5_posthoc": all5["worst_risk_seen"],
        "worst_risk_unseen_all5_posthoc": all5["worst_risk_unseen"],
        "worst_acc_all5_posthoc": all5["worst_acc"],
    }
    for hospital in range(5):
        row[f"acc_h{hospital}"] = all5["acc"][hospital]
        row[f"risk_h{hospital}"] = all5["risk"][hospital]
        if hospital in val4["risk"]:
            row[f"acc_val_h{hospital}"] = val4["acc"][hospital]
            row[f"risk_val_h{hospital}"] = val4["risk"][hospital]
    # Giữ tên cột cũ để các notebook/bảng cũ không bị lỗi; ghi rõ đây là all-5 post-hoc.
    row["avg_risk"] = row["avg_risk_all5_posthoc"]
    row["worst_risk"] = row["worst_risk_all5_posthoc"]
    row["worst_risk_seen"] = row["worst_risk_seen_all5_posthoc"]
    row["worst_risk_unseen"] = row["worst_risk_unseen_all5_posthoc"]
    row["worst_acc"] = row["worst_acc_all5_posthoc"]
    return row


def run_experiment(
    run_name, seed, args, device, splits, hospital_idx, n_groups, results_dir,
    train_groups, evaluate_only=False,
):
    set_seed(seed)
    tag = f"{run_name}_seed{seed}"
    best_path = results_dir / f"{tag}_best.pth"
    latest_path = results_dir / f"{tag}_latest.pt"
    history_path = results_dir / f"{tag}_history.json"

    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    group_weights = torch.zeros(n_groups, device=device)
    group_weights[train_groups] = 1.0 / len(train_groups)
    history = {"train_loss": [], "id_val": [], "ood_val": []}
    best_id_acc = -1.0
    start_epoch = 1

    expected_run_config = {
        "method": EXPERIMENT_CONFIGS[run_name]["method"],
        "augmentation": EXPERIMENT_CONFIGS[run_name]["augmentation"],
        "batch_size": args.batch_size,
        "lr": args.lr,
        "training_impl_version": 2,
    }

    if evaluate_only:
        source = Path(args.checkpoint_dir) / f"{tag}_best.pth"
        if not source.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint để đánh giá: {source}")
        model.load_state_dict(torch.load(source, map_location=device, weights_only=True))
        history_source = Path(args.checkpoint_dir) / f"{tag}_history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        elif history_source.exists():
            history = json.loads(history_source.read_text(encoding="utf-8"))
        print(f"[evaluate-only] Đã nạp {source}")
    else:
        if args.resume and latest_path.exists():
            checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
            saved_config = checkpoint.get("run_config")
            if saved_config != expected_run_config:
                raise RuntimeError(
                    f"Checkpoint {latest_path} không có run_config tương thích. "
                    "Không resume checkpoint cũ/khác augmentation hoặc khác GroupDRO implementation; "
                    "hãy giữ checkpoint cũ làm legacy và bắt đầu run corrected mới."
                )
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            history = checkpoint.get("history", history)
            best_id_acc = checkpoint.get("best_id_val_acc", -1.0)
            start_epoch = int(checkpoint["epoch"]) + 1
            if run_name.startswith("group_dro"):
                saved_weights = checkpoint.get("group_weights")
                if saved_weights is None or saved_weights.numel() != n_groups:
                    raise ValueError("Checkpoint GroupDRO không có group_weights tương thích.")
                group_weights = saved_weights.to(device)
                inactive = torch.ones(n_groups, dtype=torch.bool, device=device)
                inactive[train_groups] = False
                group_weights[inactive] = 0
                if group_weights.sum() <= 0:
                    group_weights[train_groups] = 1.0 / len(train_groups)
                else:
                    group_weights /= group_weights.sum()
            print(f"[resume] Khôi phục sau epoch {start_epoch - 1}; epoch tiếp theo={start_epoch}")

        if start_epoch <= args.epochs:
            for epoch in range(start_epoch, args.epochs + 1):
                if run_name.startswith("group_dro"):
                    train_loss = train_one_epoch_group_dro(
                        model, splits["train"], optimizer, device, n_groups,
                        hospital_idx, group_weights,
                    )
                else:
                    train_loss = train_one_epoch_erm(model, splits["train"], optimizer, device)

                id_val = evaluate(model, splits["id_val"], device, n_groups, hospital_idx)
                ood_val = evaluate(model, splits["val"], device, n_groups, hospital_idx)
                history["train_loss"].append(train_loss)
                history["id_val"].append(id_val)
                history["ood_val"].append(ood_val)

                # Chỉ ID validation được dùng chọn checkpoint; không dùng OOD test.
                if id_val["accuracy"] > best_id_acc:
                    best_id_acc = id_val["accuracy"]
                    atomic_torch_save(model.state_dict(), best_path)

                atomic_torch_save({
                    "epoch": epoch,
                    "run_config": expected_run_config,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "group_weights": group_weights.detach().cpu() if run_name.startswith("group_dro") else None,
                    "best_id_val_acc": best_id_acc,
                    "history": history,
                    "train_groups": train_groups.detach().cpu(),
                }, latest_path)
                print(
                    f"{tag} epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} "
                    f"id_acc={id_val['accuracy']:.4f} ood_val_acc={ood_val['accuracy']:.4f}"
                )
        else:
            print(f"[resume] {tag} đã đủ {args.epochs} epochs; bỏ qua training.")
        if not best_path.exists():
            raise FileNotFoundError(
                f"Không có best checkpoint: {best_path}. Không thể đánh giá an toàn."
            )
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    parts = {
        "id_val": evaluate(model, splits["id_val"], device, n_groups, hospital_idx),
        "val": evaluate(model, splits["val"], device, n_groups, hospital_idx),
        "test": evaluate(model, splits["test"], device, n_groups, hospital_idx),
    }
    val4 = evaluate_validation_domains(
        model, splits, device, n_groups, hospital_idx, seen=train_groups.tolist(), parts=parts
    )
    all5 = evaluate_all_hospitals(
        model, splits, device, n_groups, hospital_idx, seen=train_groups.tolist(), parts=parts
    )
    history_path.write_text(json.dumps(json_safe(history), indent=2), encoding="utf-8")
    row = flatten_metrics(run_name, seed, parts, val4, all5)
    print(
        f"[{tag}] VALIDATION-4 avg_risk={val4['avg_risk']:.4f}, "
        f"worst={val4['worst_risk']:.4f}; "
        f"ALL-5 posthoc avg={all5['avg_risk']:.4f}, worst={all5['worst_risk']:.4f}"
    )
    return row


def main():
    parser = argparse.ArgumentParser(description="Camelyon17-WILDS multi-hospital experiments")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--runs", nargs="+", choices=list(EXPERIMENT_CONFIGS), default=["erm"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--archive-path", default=None)
    parser.add_argument("--num-workers", type=int, default=2 if os.name != "nt" else 0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results-dir", default="results/corrected_v1")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs và --batch-size phải >= 1.")
    if args.evaluate_only and not args.checkpoint_dir:
        parser.error("--evaluate-only yêu cầu --checkpoint-dir.")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | results: {results_dir.resolve()}")

    # Khởi tạo split baseline một lần; dùng làm nguồn domain stats và group mapping.
    base_splits, hospital_idx, n_groups = get_dataloaders(
        batch_size=args.batch_size, use_augmentation=False, data_dir=args.data_dir,
        num_workers=args.num_workers, archive_path=args.archive_path,
    )
    train_groups = torch.unique(
        torch.as_tensor(base_splits["train"].dataset.metadata_array)[:, hospital_idx].long()
    ).to(device)
    if len(train_groups) == 0:
        raise RuntimeError("Không tìm thấy hospital trong split train.")

    stats_path = results_dir / "domain_stats.csv"
    if not stats_path.exists():
        get_domain_stats(base_splits, hospital_idx).to_csv(stats_path, index=False)

    # Tạo DataLoader augmentation riêng đúng một lần nếu có run cần augmentation.
    splits_by_augmentation = {False: base_splits}
    if not args.evaluate_only and any(EXPERIMENT_CONFIGS[r]["augmentation"] for r in args.runs):
        augmented_splits, aug_hospital_idx, aug_n_groups = get_dataloaders(
            batch_size=args.batch_size, use_augmentation=True, data_dir=args.data_dir,
            num_workers=args.num_workers, archive_path=args.archive_path,
        )
        if aug_hospital_idx != hospital_idx or aug_n_groups != n_groups:
            raise RuntimeError("Metadata mapping giữa hai pipeline không nhất quán.")
        splits_by_augmentation[True] = augmented_splits

    rows = []
    for run_name in args.runs:
        config = EXPERIMENT_CONFIGS[run_name]
        run_splits = splits_by_augmentation.get(config["augmentation"], base_splits)
        for seed in args.seeds:
            rows.append(run_experiment(
                run_name, seed, args, device, run_splits, hospital_idx, n_groups,
                results_dir, train_groups, evaluate_only=args.evaluate_only,
            ))

    new_df = pd.DataFrame(rows)
    csv_path = results_dir / "metrics.csv"
    if csv_path.exists() and csv_path.stat().st_size:
        old_df = pd.read_csv(csv_path)
        df = pd.concat([old_df, new_df], ignore_index=True)
        df = df.drop_duplicates(subset=["model", "seed"], keep="last")
    else:
        df = new_df
    df.to_csv(csv_path, index=False)
    print(f"\nMetrics: {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
