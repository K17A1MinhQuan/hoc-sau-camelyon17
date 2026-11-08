"""Dataset Camelyon17-WILDS và thống kê domain."""
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from wilds import get_dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_train_transform(use_augmentation=False):
    if not use_augmentation:
        return get_eval_transform()
    return transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _prepare_archive(data_dir, archive_path=None):
    """Đặt archive vào vị trí WILDS mong đợi nếu đã có archive bên ngoài."""
    data_root = Path(data_dir).expanduser().resolve()
    expected_dir = data_root / "camelyon17_v1.0"
    expected_archive = expected_dir / "archive.tar.gz"

    if (expected_dir / "metadata.csv").exists() and (expected_dir / "patches").exists():
        print(f"[dataset] Dùng dữ liệu đã giải nén: {expected_dir}")
        return

    candidates = []
    if archive_path:
        candidates.append(Path(archive_path).expanduser())
    candidates.extend([
        data_root / "camelyon17_v1.0.tar.gz",
        data_root / "archive.tar.gz",
        Path("/content/drive/MyDrive/camelyon17_v1.0.tar.gz"),
    ])

    for candidate in candidates:
        if candidate.is_file():
            expected_dir.mkdir(parents=True, exist_ok=True)
            if candidate.resolve() != expected_archive.resolve():
                try:
                    if expected_archive.exists() or expected_archive.is_symlink():
                        expected_archive.unlink()
                    expected_archive.symlink_to(candidate.resolve())
                except OSError:
                    print(f"[dataset] Không thể tạo symlink cho {candidate}.")
                    print(f"[dataset] Hãy đặt archive vào: {expected_archive}")
                    continue
            print(f"[dataset] Archive có sẵn: {candidate}")
            return

    print("[dataset] Không tìm thấy archive ở các đường dẫn đã kiểm tra.")
    print("[dataset] WILDS có thể tải dữ liệu nếu download=True; hãy kiểm tra dung lượng.")


def get_domain_stats(splits, hospital_idx, target_idx):
    """Thống kê số mẫu và tỷ lệ tumor theo split/hospital."""
    rows = []
    for split_name, loader in splits.items():
        metadata = torch.as_tensor(loader.dataset.metadata_array)
        if metadata.ndim != 2:
            raise ValueError(f"metadata_array của split {split_name} không phải ma trận 2D.")

        hospital_values = metadata[:, hospital_idx].long()
        targets = metadata[:, target_idx].float()
        for hospital in torch.unique(hospital_values).tolist():
            mask = hospital_values == int(hospital)
            rows.append({
                "split": split_name,
                "hospital": int(hospital),
                "n_samples": int(mask.sum().item()),
                "tumor_ratio": float(targets[mask].mean().item()),
                "n_tumor": int(targets[mask].sum().item()),
                "n_non_tumor": int((~targets[mask].bool()).sum().item()),
            })
    return pd.DataFrame(rows)


def get_dataloaders(
    batch_size=16,
    use_augmentation=False,
    data_dir="data",
    num_workers=0,
    archive_path=None,
    download=True,
):
    """Trả về splits, chỉ số metadata hospital, số hospital và chỉ số target."""
    _prepare_archive(data_dir, archive_path)
    dataset = get_dataset(
        dataset="camelyon17", root_dir=str(Path(data_dir).expanduser()), download=download
    )

    fields = list(dataset.metadata_fields)
    if "hospital" not in fields:
        raise ValueError(f"Không tìm thấy metadata field 'hospital': {fields}")
    if "y" not in fields:
        raise ValueError(f"Không tìm thấy metadata field 'y': {fields}")

    hospital_idx = fields.index("hospital")
    target_idx = fields.index("y")
    n_groups = 5

    splits = {}
    split_names = {"train": "train", "id_val": "id_val", "val": "val", "test": "test"}
    train_transform = get_train_transform(use_augmentation)
    eval_transform = get_eval_transform()

    for key, split_name in split_names.items():
        subset = dataset.get_subset(
            split_name, transform=train_transform if key == "train" else eval_transform
        )
        if len(subset) == 0:
            raise RuntimeError(f"Split '{split_name}' rỗng; kiểm tra dataset và metadata.")
        splits[key] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(key == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
        )

    return splits, hospital_idx, n_groups, target_idx
