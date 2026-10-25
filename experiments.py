"""experiments.py — Điều phối các thí nghiệm.

Chạy: python experiments.py --epochs 5 --batch-size 16 --runs erm erm_aug group_dro group_dro_aug
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import get_dataloaders
from model import get_model
from train import train_one_epoch_erm, train_one_epoch_group_dro
from evaluate import evaluate


RESULTS_DIR = "results"

# Cấu hình thí nghiệm
EXPERIMENT_CONFIGS = {
    "erm":           {"method": "erm",       "augmentation": False},
    "erm_aug":       {"method": "erm",       "augmentation": True},
    "group_dro":     {"method": "group_dro", "augmentation": False},
    "group_dro_aug": {"method": "group_dro", "augmentation": True},
}


def set_seed(seed):
    """Đặt random seed cho reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def run_experiment(run_name, config, seed, epochs, batch_size, lr, data_dir, device, resume=False, custom_start_epoch=None, num_workers=0, archive_path=None):
    """Chạy một thí nghiệm đầy đủ: train → validate → test."""
    print(f"\n{'='*60}")
    print(f"  {run_name}  |  seed={seed}  |  epochs={epochs}  |  device={device}")
    print(f"{'='*60}")

    set_seed(seed)

    # Data
    splits, hospital_idx, n_groups = get_dataloaders(
        batch_size=batch_size,
        use_augmentation=config["augmentation"],
        data_dir=data_dir,
        num_workers=num_workers,
        archive_path=archive_path,
    )

    # Model & Optimizer
    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Group DRO weights
    group_weights = torch.ones(n_groups, device=device) / n_groups

    # Training history
    history = {"train_loss": [], "id_val": [], "ood_val": []}
    best_id_val_acc = -1.0
    start_epoch = 1
    tag = f"{run_name}_seed{seed}"
    checkpoint_path = os.path.join(RESULTS_DIR, f"{tag}_best.pth")
    latest_path = os.path.join(RESULTS_DIR, f"{tag}_latest.pt")

    # Resume nếu có yêu cầu và checkpoint tồn tại
    if resume and os.path.exists(latest_path):
        print(f"  → Tìm thấy checkpoint: {latest_path}. Đang tải lại...")
        try:
            ckpt = torch.load(latest_path, map_location=device, weights_only=False)
            required = {"epoch", "model_state_dict", "optimizer_state_dict"}
            if not isinstance(ckpt, dict) or not required.issubset(ckpt):
                raise ValueError("Checkpoint thiếu trường bắt buộc.")
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if config["method"] == "group_dro" and ckpt.get("group_weights") is not None:
                group_weights = ckpt["group_weights"].to(device)
            best_id_val_acc = ckpt.get("best_id_val_acc", -1.0)
            history = ckpt.get("history", history)
            start_epoch = int(ckpt["epoch"]) + 1
            print(f"  ✓ Đã phục hồi trạng thái sau Epoch {ckpt['epoch']}. Bắt đầu từ Epoch {start_epoch}/{epochs}.")
        except Exception as exc:
            print(f"  ! Checkpoint latest bị lỗi/không tương thích: {exc}")
            print("  ! Bỏ qua latest và thử tìm best checkpoint. Hãy sao lưu file lỗi để kiểm tra.")
            resume = True
            # Đổi tên file lỗi để fallback hoạt động; vẫn giữ file để kiểm tra sau.
            try:
                os.replace(latest_path, latest_path + ".corrupt")
                print(f"  ! Đã giữ checkpoint lỗi tại {latest_path}.corrupt")
            except OSError as rename_exc:
                print(f"  ! Không đổi tên được checkpoint lỗi: {rename_exc}")
    if resume and (not os.path.exists(latest_path) if latest_path else True):
        # Fallback: kiểm tra file _best.pth ở results/ hoặc thư mục gốc
        best_candidate = None
        for p in [checkpoint_path, f"{tag}_best.pth"]:
            if os.path.exists(p):
                best_candidate = p
                break
        if best_candidate:
            print(f"  → Không thấy {latest_path}, tải trọng số từ best checkpoint: {best_candidate}...")
            try:
                ckpt = torch.load(best_candidate, map_location=device, weights_only=True)
                model.load_state_dict(ckpt)
                print(f"  ✓ Đã nạp xong trọng số mô hình từ {best_candidate}.")
            except Exception as exc:
                print(f"  ! Best checkpoint cũng không đọc được: {exc}. Sẽ khởi tạo mô hình mới.")
                best_candidate = None

            if not best_candidate:
                start_epoch = 1
            # Thử nạp lại history nếu có sẵn để giữ biểu đồ liền mạch
            history_candidate = None
            for hp in [os.path.join(RESULTS_DIR, f"{tag}_history.json"), f"{tag}_history.json"]:
                if os.path.exists(hp):
                    history_candidate = hp
                    break
            if history_candidate:
                try:
                    with open(history_candidate, "r") as f:
                        loaded_history = json.load(f)
                    history = loaded_history
                    n_done = len(history.get("train_loss", []))
                    if n_done > 0:
                        start_epoch = n_done + 1
                        print(f"  ✓ Tự động tìm thấy {history_candidate} (đã chạy {n_done} epochs). Tiếp tục từ Epoch {start_epoch}!")
                except Exception:
                    pass
        else:
            print(f"  ! Không tìm thấy checkpoint nào để resume cho {tag}.")

    # Cho phép ghi đè start_epoch nếu người dùng chỉ định cụ thể qua tham số
    if custom_start_epoch is not None:
        start_epoch = custom_start_epoch
        print(f"  → Chỉ định bắt đầu từ Epoch {start_epoch} (theo tham số --start-epoch).")

    try:
        if start_epoch > epochs:
            print(f"  → Mô hình đã hoàn thành đủ {epochs} epochs trước đó. Bỏ qua huấn luyện.")
        else:
            for epoch in range(start_epoch, epochs + 1):
                print(f"\nEpoch {epoch}/{epochs}")

                # Train
                if config["method"] == "erm":
                    train_loss = train_one_epoch_erm(model, splits["train"], optimizer, device)
                else:
                    train_loss = train_one_epoch_group_dro(
                        model, splits["train"], optimizer, device,
                        n_groups, hospital_idx, group_weights,
                    )
                history["train_loss"].append(train_loss)
                print(f"  Train loss: {train_loss:.4f}")

                # ID validation
                id_val_metrics = evaluate(model, splits["id_val"], device, n_groups, hospital_idx)
                history["id_val"].append(id_val_metrics)
                print(f"  ID Val  — Acc: {id_val_metrics['accuracy']:.4f}  AUROC: {id_val_metrics['auroc']:.4f}")

                # OOD validation
                ood_val_metrics = evaluate(model, splits["val"], device, n_groups, hospital_idx)
                history["ood_val"].append(ood_val_metrics)
                print(f"  OOD Val — Acc: {ood_val_metrics['accuracy']:.4f}  AUROC: {ood_val_metrics['auroc']:.4f}")

                # Checkpoint selection: best ID validation accuracy
                if id_val_metrics["accuracy"] > best_id_val_acc:
                    best_id_val_acc = id_val_metrics["accuracy"]
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"  ✓ Saved best checkpoint (ID val acc={best_id_val_acc:.4f})")

                # Lưu latest checkpoint sau mỗi epoch để hỗ trợ resume
                latest_ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "group_weights": group_weights.cpu() if config["method"] == "group_dro" else None,
                    "best_id_val_acc": best_id_val_acc,
                    "history": history,
                }
                torch.save(latest_ckpt, latest_path)
                print(f"  ✓ Đã lưu checkpoint epoch {epoch} vào {latest_path}")

                # Giải phóng bộ nhớ GPU đệm sau mỗi epoch
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n\n  [Cảnh báo] Huấn luyện bị ngắt bởi người dùng. Đang lưu trạng thái khẩn cấp...")
        latest_ckpt = {
            "epoch": epoch if 'epoch' in locals() else start_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "group_weights": group_weights.cpu() if config["method"] == "group_dro" else None,
            "best_id_val_acc": best_id_val_acc,
            "history": history,
        }
        torch.save(latest_ckpt, latest_path)
        print(f"  ✓ Đã lưu trạng thái an toàn vào {latest_path}. Bạn có thể tiếp tục với --resume bất cứ lúc nào.")
        raise

    # Load best checkpoint for final test (nếu có, không thì dùng trọng số hiện tại)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    elif os.path.exists(f"{tag}_best.pth"):
        model.load_state_dict(torch.load(f"{tag}_best.pth", map_location=device, weights_only=True))

    print(f"\nFinal test evaluation ({tag}):")
    id_val_final = evaluate(model, splits["id_val"], device, n_groups, hospital_idx)
    ood_test = evaluate(model, splits["test"], device, n_groups, hospital_idx)

    print(f"  ID Val  — Acc: {id_val_final['accuracy']:.4f}  AUROC: {id_val_final['auroc']:.4f}")
    print(f"  OOD Test— Acc: {ood_test['accuracy']:.4f}  AUROC: {ood_test['auroc']:.4f}")
    print(f"  Avg Group Risk: {ood_test['avg_group_risk']:.4f}")
    print(f"  Worst Group Risk: {ood_test['worst_group_risk']:.4f}")
    print(f"  Per-group Acc: {ood_test['per_group_acc']}")

    # Save history
    history_path = os.path.join(RESULTS_DIR, f"{tag}_history.json")
    # Convert numpy/float values for JSON serialization
    def _serialize(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(i) for i in obj]
        return obj

    with open(history_path, "w") as f:
        json.dump(_serialize(history), f, indent=2)

    return {
        "model": run_name,
        "seed": seed,
        "id_accuracy": id_val_final["accuracy"],
        "ood_accuracy": ood_test["accuracy"],
        "id_auroc": id_val_final["auroc"],
        "ood_auroc": ood_test["auroc"],
        "avg_group_risk": ood_test["avg_group_risk"],
        "worst_group_risk": ood_test["worst_group_risk"],
    }, history


def plot_results(df, all_histories):
    """Tạo 3 biểu đồ theo yêu cầu README mục 9."""
    if df.empty:
        return

    # 1. So sánh ID và OOD performance
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models = df.groupby("model")

    model_names = []
    id_accs = []
    ood_accs = []
    id_aurocs = []
    ood_aurocs = []
    for name, group in models:
        model_names.append(name)
        id_accs.append(group["id_accuracy"].mean())
        ood_accs.append(group["ood_accuracy"].mean())
        id_aurocs.append(group["id_auroc"].mean())
        ood_aurocs.append(group["ood_auroc"].mean())

    x = np.arange(len(model_names))
    w = 0.35

    axes[0].bar(x - w/2, id_accs, w, label="ID Accuracy", color="#4285F4")
    axes[0].bar(x + w/2, ood_accs, w, label="OOD Accuracy", color="#EA4335")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=30, ha="right")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("ID vs OOD Accuracy")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    axes[1].bar(x - w/2, id_aurocs, w, label="ID AUROC", color="#4285F4")
    axes[1].bar(x + w/2, ood_aurocs, w, label="OOD AUROC", color="#EA4335")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=30, ha="right")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("ID vs OOD AUROC")
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_id_ood.png"), dpi=150)
    plt.close()

    # 2. Training loss và validation loss theo epoch
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4285F4", "#EA4335", "#FBBC04", "#34A853"]
    for i, (tag, hist) in enumerate(all_histories.items()):
        epochs = range(1, len(hist["train_loss"]) + 1)
        c = colors[i % len(colors)]
        ax.plot(epochs, hist["train_loss"], "-", color=c, label=f"{tag} train")
        val_losses = [m["loss"] for m in hist["ood_val"]]
        ax.plot(epochs, val_losses, "--", color=c, label=f"{tag} OOD val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & OOD Validation Loss")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curves.png"), dpi=150)
    plt.close()

    # 3. OOD Test Accuracy per Experiment
    fig, ax = plt.subplots(figsize=(10, 6))
    model_grouped = df.groupby("model")["ood_accuracy"].mean()
    model_names_2 = list(model_grouped.index)
    ood_accs_2 = list(model_grouped.values)
    ax.bar(range(len(model_names_2)), ood_accs_2, color=colors[:len(model_names_2)])
    ax.set_xticks(range(len(model_names_2)))
    ax.set_xticklabels(model_names_2, rotation=30, ha="right")
    ax.set_ylabel("OOD Test Accuracy")
    ax.set_title("OOD Test Accuracy per Experiment")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "per_experiment_accuracy.png"), dpi=150)
    plt.close()

    print(f"\nCharts saved to {RESULTS_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Camelyon17-WILDS Experiments")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--runs", type=str, nargs="+", default=["erm"],
                        choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--archive-path", type=str, default=None,
                        help="Đường dẫn archive Camelyon17 .tar.gz có sẵn (ví dụ trên Google Drive)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint if available")
    parser.add_argument("--start-epoch", type=int, default=None,
                        help="Chỉ định epoch bắt đầu khi tiếp tục huấn luyện (ví dụ: 7)")
    default_workers = 2 if os.name != "nt" else 0
    parser.add_argument("--num-workers", type=int, default=default_workers,
                        help=f"Số worker tải dữ liệu song song (mặc định: {default_workers})")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Data workers: {args.num_workers}")

    all_results = []
    all_histories = {}

    for run_name in args.runs:
        config = EXPERIMENT_CONFIGS[run_name]
        for seed in args.seeds:
            result, history = run_experiment(
                run_name, config, seed, args.epochs, args.batch_size,
                args.lr, args.data_dir, device, resume=args.resume,
                custom_start_epoch=args.start_epoch,
                num_workers=args.num_workers,
                archive_path=args.archive_path,
            )
            all_results.append(result)
            all_histories[f"{run_name}_seed{seed}"] = history

    # Gộp kết quả với metrics.csv cũ nếu có để tránh mất dữ liệu các lần chạy trước
    csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    new_df = pd.DataFrame(all_results)
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        old_df = pd.read_csv(csv_path)
        df = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(subset=["model", "seed"], keep="last")
    else:
        df = new_df

    df.to_csv(csv_path, index=False)
    print(f"\n{'='*60}")
    print("Summary:")
    print(df.to_string(index=False))
    print(f"\nResults saved to {csv_path}")

    # Phân tích Q1, Q2, Q3
    print(f"\n{'='*60}")
    print("Analysis")
    print(f"{'='*60}")

    # Q1: Performance loss do domain shift
    erm_rows = df[df["model"] == "erm"]
    if not erm_rows.empty:
        id_acc = erm_rows["id_accuracy"].mean()
        ood_acc = erm_rows["ood_accuracy"].mean()
        gap = id_acc - ood_acc
        relative = (gap / id_acc * 100) if id_acc > 0 else float("nan")
        print(f"\nQ1 — Domain Shift Gap (ERM baseline):")
        print(f"  ID Accuracy:  {id_acc:.4f}")
        print(f"  OOD Accuracy: {ood_acc:.4f}")
        print(f"  Gap:          {gap:.4f}")
        print(f"  Relative:     {relative:.1f}%")

    # Q2: Data Augmentation
    erm_aug_rows = df[df["model"] == "erm_aug"]
    if not erm_rows.empty and not erm_aug_rows.empty:
        delta = erm_aug_rows["ood_accuracy"].mean() - erm_rows["ood_accuracy"].mean()
        print(f"\nQ2 — Data Augmentation Effect:")
        print(f"  ΔOOD Accuracy (aug - baseline): {delta:+.4f}")

    # Q3: ERM vs Group DRO
    dro_rows = df[df["model"] == "group_dro"]
    if not erm_rows.empty and not dro_rows.empty:
        print(f"\nQ3 — ERM vs Group DRO:")
        print(f"  ERM  — Avg Risk: {erm_rows['avg_group_risk'].mean():.4f}  "
              f"Worst Risk: {erm_rows['worst_group_risk'].mean():.4f}")
        print(f"  DRO  — Avg Risk: {dro_rows['avg_group_risk'].mean():.4f}  "
              f"Worst Risk: {dro_rows['worst_group_risk'].mean():.4f}")

    # Tải thêm các history cũ từ results/ (nếu có) để vẽ biểu đồ loss đầy đủ các run
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith("_history.json"):
            tag = fname.replace("_history.json", "")
            if tag not in all_histories:
                try:
                    with open(os.path.join(RESULTS_DIR, fname), "r") as f:
                        all_histories[tag] = json.load(f)
                except Exception:
                    pass

    # Vẽ biểu đồ tổng hợp
    plot_results(df, all_histories)


if __name__ == "__main__":
    main()
