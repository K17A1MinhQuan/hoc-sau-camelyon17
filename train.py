"""Một epoch huấn luyện ERM và Group DRO."""
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

def train_one_epoch_group_dro(model, dataloader, optimizer, device,
                              n_groups, hospital_idx, group_weights,
                              dro_step_size=0.01):
    """Group DRO ổn định: chỉ cập nhật loss/weight của nhóm xuất hiện trong batch."""
    model.train()
    total_loss, n_samples = 0.0, 0
    group_weights = group_weights.to(device)
    for x, y, metadata in tqdm(dataloader, desc="Training (Group DRO)", leave=False):
        x, y = x.to(device), y.to(device).long()
        groups = metadata[:, hospital_idx].to(device).long()
        if ((groups < 0) | (groups >= n_groups)).any():
            raise ValueError(f"Group ID ngoài phạm vi [0, {n_groups-1}]. Kiểm tra metadata.")
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        losses = F.cross_entropy(logits, y, reduction="none")
        present_ids = torch.unique(groups)
        batch_group_losses = []
        for group_id in present_ids:
            mask = groups == group_id
            group_loss = losses[mask].mean()
            batch_group_losses.append((group_id, group_loss))
        # Multiplicative update chỉ cho nhóm hiện diện, rồi chuẩn hóa.
        with torch.no_grad():
            for group_id, group_loss in batch_group_losses:
                group_weights[group_id] *= torch.exp(dro_step_size * group_loss.detach())
            total_weight = group_weights.sum()
            if not torch.isfinite(total_weight) or total_weight <= 0:
                group_weights.fill_(1.0 / n_groups)
            else:
                group_weights /= total_weight
        loss = sum(group_weights[g] * gl for g, gl in batch_group_losses)
        if not torch.isfinite(loss):
            raise RuntimeError("Group DRO loss không hữu hạn; dừng để tránh checkpoint lỗi.")
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() * x.size(0)
        n_samples += x.size(0)
    if n_samples == 0:
        raise RuntimeError("DataLoader train rỗng.")
    return total_loss / n_samples
