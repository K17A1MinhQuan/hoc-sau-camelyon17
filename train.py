"""Một epoch huấn luyện ERM và GroupDRO."""
import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_one_epoch_erm(model, dataloader, optimizer, device):
    model.train()
    total_loss, n_samples = 0.0, 0
    for x, y, metadata in tqdm(dataloader, desc="Training (ERM)", leave=False):
        x, y = x.to(device), y.to(device).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() * x.size(0)
        n_samples += x.size(0)
    if n_samples == 0:
        raise RuntimeError("DataLoader train rỗng.")
    return total_loss / n_samples


def train_one_epoch_group_dro(
    model, dataloader, optimizer, device, n_groups, hospital_idx,
    group_weights, dro_step_size=0.01,
):
    """GroupDRO minibatch approximation với chuẩn hóa trên nhóm hiện diện.

    q được cập nhật multiplicative trên các nhóm active đã xuất hiện trong batch.
    Loss của batch được chuẩn hóa lại trên các nhóm hiện diện để tránh làm nhỏ
    gradient chỉ vì một số nhóm active không có mặt trong minibatch.
    Đây là stochastic minibatch approximation; không phải full-batch GroupDRO.
    """
    model.train()
    total_loss, n_samples = 0.0, 0
    group_weights = group_weights.to(device)

    active_mask = group_weights > 0
    if not active_mask.any():
        raise ValueError("GroupDRO cần ít nhất một nhóm training có trọng số dương.")

    for x, y, metadata in tqdm(dataloader, desc="Training (GroupDRO)", leave=False):
        x, y = x.to(device), y.to(device).long()
        groups = metadata[:, hospital_idx].to(device).long()
        if ((groups < 0) | (groups >= n_groups)).any():
            raise ValueError(f"Group ID ngoài phạm vi [0, {n_groups - 1}].")
        if (~active_mask[groups]).any():
            raise ValueError("Batch chứa hospital không thuộc các nhóm training active.")

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        losses = F.cross_entropy(logits, y, reduction="none")
        present_ids = torch.unique(groups)
        batch_group_losses = torch.stack([
            losses[groups == group_id].mean() for group_id in present_ids
        ])

        # Multiplicative-weights update, giữ q chuẩn hóa trên toàn bộ nhóm active.
        with torch.no_grad():
            group_weights[present_ids] *= torch.exp(
                dro_step_size * batch_group_losses.detach()
            )
            group_weights[~active_mask] = 0.0
            total_weight = group_weights[active_mask].sum()
            if not torch.isfinite(total_weight) or total_weight <= 0:
                raise RuntimeError("GroupDRO weights không hợp lệ.")
            group_weights[active_mask] /= total_weight

        # Chỉ các nhóm có mặt mới có loss ước lượng ở batch này.
        present_weights = group_weights[present_ids]
        present_weights = present_weights / present_weights.sum().clamp_min(1e-12)
        loss = torch.sum(present_weights * batch_group_losses)
        if not torch.isfinite(loss):
            raise RuntimeError("GroupDRO loss không hữu hạn.")
        loss.backward()
        optimizer.step()

        total_loss += losses.detach().mean().item() * x.size(0)
        n_samples += x.size(0)

    if n_samples == 0:
        raise RuntimeError("DataLoader train rỗng.")
    return total_loss / n_samples
