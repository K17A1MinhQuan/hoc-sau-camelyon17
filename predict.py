"""predict.py — Tải mô hình đã huấn luyện và dự đoán nhãn cho một ảnh mới.

Cách chạy:
    1. Chạy với ảnh cụ thể:
       python predict.py --image "duong_dan_anh.png"

    2. Tự động lấy một ảnh mẫu có sẵn trong thư mục data:
       python predict.py

    3. Chỉ định checkpoint cụ thể:
       python predict.py --image "duong_dan_anh.png" --checkpoint "results/erm_seed42_best.pth"
"""

import argparse
import glob
import os
import sys
from PIL import Image

import torch
import torch.nn.functional as F

from dataset import get_eval_transform
from model import get_model


# Định nghĩa nhãn của bài toán Camelyon17
CLASS_NAMES = {
    0: "Bình thường / Không có khối u (Negative - Normal)",
    1: "Khối u di căn (Positive - Tumor)",
}


def find_default_checkpoint(results_dir="results"):
    """Tự động tìm checkpoint .pth tốt nhất có sẵn trong thư mục results."""
    # Ưu tiên các checkpoint có đuôi _best.pth
    best_checkpoints = glob.glob(os.path.join(results_dir, "*_best.pth"))
    if best_checkpoints:
        return best_checkpoints[0]

    # Nếu không có _best.pth, tìm bất kỳ file .pth nào
    all_checkpoints = glob.glob(os.path.join(results_dir, "*.pth"))
    if all_checkpoints:
        return all_checkpoints[0]

    return None


def find_sample_image(data_dir="data"):
    """Tự động tìm 1 ảnh mẫu có sẵn trong thư mục data nếu người dùng chưa chỉ định."""
    pattern = os.path.join(data_dir, "**", "*.png")
    sample_images = glob.glob(pattern, recursive=True)
    if sample_images:
        return sample_images[0]
    return None


def predict_image(image_path, checkpoint_path, device=None):
    """Thực hiện dự đoán nhãn cho 1 ảnh."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device sử dụng: {device}")

    # 1. Kiểm tra sự tồn tại của file checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"\n[LỖI] Không tìm thấy file checkpoint tại: '{checkpoint_path}'")
        print("Vui lòng đợi quá trình huấn luyện hoàn tất hoặc kiểm tra lại thư mục 'results/'.")
        sys.exit(1)

    # 2. Kiểm tra sự tồn tại của file ảnh
    if not os.path.exists(image_path):
        print(f"\n[LỖI] Không tìm thấy file ảnh tại: '{image_path}'")
        sys.exit(1)

    print(f"\nĐang tải checkpoint: {checkpoint_path}")
    # 3. Khởi tạo và nạp trọng số mô hình
    model = get_model(num_classes=2).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Tiền xử lý ảnh (Resize 96x96, ImageNet Normalization)
    print(f"Đang xử lý ảnh: {image_path}")
    try:
        raw_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[LỖI] Không thể đọc file ảnh: {e}")
        sys.exit(1)

    transform = get_eval_transform()
    input_tensor = transform(raw_image).unsqueeze(0).to(device)  # Shape: (1, 3, 96, 96)

    # 5. Dự đoán
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)  # Shape: (2,)
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item() * 100

    # 6. Hiển thị kết quả
    print("\n" + "=" * 60)
    print("                     KẾT QUẢ DỰ ĐOÁN")
    print("=" * 60)
    print(f"Ảnh đầu vào        : {os.path.basename(image_path)}")
    print(f"Lớp dự đoán (Class): {pred_class}")
    print(f"Nhãn chẩn đoán     : {CLASS_NAMES.get(pred_class, 'Chưa xác định')}")
    print(f"Độ tin cậy         : {confidence:.2f}%")
    print("-" * 60)
    print("Chi tiết xác suất từng lớp:")
    for cls_idx, cls_name in CLASS_NAMES.items():
        prob_pct = probs[cls_idx].item() * 100
        bar = "█" * int(prob_pct // 4)
        print(f"  Class {cls_idx} ({prob_pct:5.2f}%): {bar:<25} [{cls_name}]")
    print("=" * 60 + "\n")

    return pred_class, probs.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Dự đoán ảnh mới bằng mô hình Camelyon17 đã huấn luyện")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Đường dẫn đến file ảnh cần test (hỗ trợ .png, .jpg, ...)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Đường dẫn tới file checkpoint .pth (mặc định tự tìm trong results/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Thiết bị chạy suy luận (mặc định: tự động chọn GPU nếu có)",
    )

    args = parser.parse_args()

    # Xác định checkpoint
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = find_default_checkpoint()
        if not checkpoint_path:
            checkpoint_path = os.path.join("results", "erm_seed42_best.pth")

    # Xác định ảnh
    image_path = args.image
    if not image_path:
        sample_img = find_sample_image()
        if sample_img:
            print(f"[Thông báo] Không có tham số --image, tự động chọn ảnh mẫu có sẵn:")
            print(f"            {sample_img}")
            image_path = sample_img
        else:
            print("[LỖI] Vui lòng truyền đường dẫn ảnh qua tham số --image, ví dụ:")
            print('      python predict.py --image "duong_dan_anh.png"')
            sys.exit(1)

    device = torch.device(args.device) if args.device else None
    predict_image(image_path, checkpoint_path, device)


if __name__ == "__main__":
    main()
