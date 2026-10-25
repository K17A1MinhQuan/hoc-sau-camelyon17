# Deep Learning Under Distribution Shift — Camelyon17-WILDS

Dự án học thuật so sánh ERM, ERM + augmentation, Group DRO và Group DRO + augmentation trên benchmark Camelyon17-WILDS.

## 1. Dataset và archive đã có

Không commit dataset, ảnh bệnh học, checkpoint hay kết quả lớn vào GitHub. Trong Google Colab, mount Drive trước:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Kiểm tra archive:

```python
from pathlib import Path
p = Path('/content/drive/MyDrive/camelyon17_v1.0.tar.gz')
print('exists:', p.is_file())
if p.is_file():
    print('GiB:', round(p.stat().st_size / 1024**3, 2))
```

Chạy từ thư mục repo. Dataset loader mặc định tìm archive tại `/content/drive/MyDrive/camelyon17_v1.0.tar.gz`, hoặc có thể truyền đường dẫn bằng `--archive-path`. Loader đặt symlink vào thư mục WILDS dự kiến để tránh sao chép archive nhiều GB. Nếu môi trường không hỗ trợ symlink, đặt archive thủ công vào `<data-dir>/camelyon17_v1.0/archive.tar.gz`. WILDS vẫn cần giải nén và kiểm tra cấu trúc dữ liệu hợp lệ; archive chỉ được nhận diện, không có nghĩa dữ liệu đã giải nén.

## 2. Cài đặt

```bash
pip install -r requirements.txt
```

Kiểm tra PyTorch/GPU:

```python
import torch
print(torch.__version__, torch.cuda.is_available())
```

## 3. Chạy thí nghiệm

```bash
python experiments.py --epochs 5 --batch-size 16 --runs erm
python experiments.py --epochs 5 --batch-size 16 --runs erm_aug
python experiments.py --epochs 5 --batch-size 16 --runs group_dro
python experiments.py --epochs 5 --batch-size 16 --runs group_dro_aug
```

Để chạy tất cả:

```bash
python experiments.py --epochs 5 --batch-size 16 --runs erm erm_aug group_dro group_dro_aug
```

Tham số hữu ích: `--data-dir data`, `--archive-path /content/drive/MyDrive/camelyon17_v1.0.tar.gz`, `--num-workers 0`, `--resume`.

## 4. Checkpoint và resume

- `*_latest.pt`: model + optimizer + group weights + epoch + history; dùng để tiếp tục đúng trạng thái.
- `*_best.pth`: chỉ có trọng số mô hình tốt nhất theo ID validation; không chứa trạng thái optimizer nên không thể khôi phục chính xác toàn bộ trạng thái huấn luyện.
- Nếu checkpoint latest bị hỏng, chương trình sẽ cảnh báo và bỏ qua file đó thay vì dừng ngay. Có thể tiếp tục từ best weights nếu file hợp lệ, hoặc bắt đầu mới.
- Không ghi checkpoint lên Google Drive trực tiếp trong lúc đang ghi nếu kết nối không ổn định. Nên lưu local trước rồi sao chép file hoàn chỉnh sang Drive.

Ví dụ tiếp tục đến epoch 10:

```bash
python experiments.py --epochs 10 --batch-size 16 --runs erm --resume
```

Chỉ resume khi cấu hình thí nghiệm (run, seed, kiến trúc) khớp checkpoint. Trước khi chạy tiếp checkpoint cũ, nên giữ một bản sao dự phòng.

## 5. Kết quả và nguyên tắc đánh giá

Các đầu ra được lưu trong `results/`: `metrics.csv`, history JSON và biểu đồ. Checkpoint tốt nhất được chọn dựa trên ID validation; OOD test chỉ dùng để đánh giá cuối cùng, không dùng để chọn mô hình hay siêu tham số.

Chỉ số AUROC được trả về NaN nếu split rỗng hoặc chỉ chứa một lớp. Các chỉ số theo nhóm chỉ tính trên những nhóm thực sự xuất hiện trong split.

## 6. Lưu ý

- Dữ liệu Camelyon17 lớn; kiểm tra dung lượng ổ đĩa trước khi giải nén.
- Không đưa dataset, checkpoint, file log hoặc thông tin nhạy cảm lên repo công khai.
- Đây là dự án nghiên cứu học thuật, không phải hệ thống chẩn đoán y tế.
- Kiểm tra giấy phép và điều khoản sử dụng của dataset trước khi phân phối dữ liệu.
