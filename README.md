# Deep Learning Under Distribution Shift: WILDS Camelyon17

## 1. Giới thiệu dự án

### 1.1. Mục tiêu

Dự án nghiên cứu ảnh hưởng của sự khác biệt giữa training distribution và test distribution đến hiệu suất của các mô hình Deep Learning.

Benchmark được sử dụng là **Camelyon17-WILDS**, một bộ dữ liệu ảnh mô bệnh học được thu thập từ nhiều bệnh viện. Mỗi bệnh viện được xem là một domain có thể có những đặc điểm dữ liệu khác nhau.

Mô hình được huấn luyện trên các domain trong tập training và được đánh giá trên dữ liệu thuộc các domain khác để kiểm tra khả năng tổng quát hóa khi phân phối dữ liệu thay đổi.

### 1.2. Các câu hỏi nghiên cứu

**Q1. How much performance is lost due to domain shift?**

Đo mức suy giảm hiệu suất khi chuyển từ dữ liệu cùng phân phối (In-Distribution — ID) sang dữ liệu ngoài phân phối (Out-of-Distribution — OOD).

**Q2. Does data augmentation improve OOD generalization?**

So sánh mô hình sử dụng data augmentation với mô hình baseline không sử dụng augmentation để xác định liệu các phép biến đổi dữ liệu có cải thiện khả năng tổng quát hóa hay không.

**Q3. Should we optimize average risk or worst-domain risk?**

So sánh Empirical Risk Minimization (ERM) và Group Distributionally Robust Optimization (Group DRO) để đánh giá sự khác biệt giữa tối ưu loss trung bình và tối ưu theo những domain có loss cao.

## 2. Dataset

### 2.1. Camelyon17-WILDS

Camelyon17-WILDS bao gồm các ảnh mô bệnh học của hạch bạch huyết, được sử dụng để phân loại các vùng mô có chứa khối u hay không.

- Input: ảnh RGB mô bệnh học.
- Kích thước ảnh gốc: 96 × 96 pixel.
- Target: nhãn nhị phân về sự hiện diện của mô khối u trong vùng trung tâm ảnh.
- Domain: bệnh viện nơi ảnh được thu thập.
- Metadata: thông tin về domain và các thuộc tính liên quan của mẫu.

Khác biệt giữa các bệnh viện có thể liên quan đến quy trình xử lý mẫu, nhuộm màu và thu nhận ảnh. Các khác biệt này tạo ra môi trường thích hợp để nghiên cứu domain shift.

Nguồn chính thức:

- Dataset: https://wilds.stanford.edu/datasets/
- GitHub: https://github.com/p-lambda/wilds
- Paper: https://proceedings.mlr.press/v139/koh21a.html

### 2.2. Training, validation và test

Dự án sử dụng các split do WILDS cung cấp.

| Split | Vai trò |
|---|---|
| `train` | Huấn luyện mô hình |
| `id_val` | Đánh giá trên dữ liệu cùng phân phối |
| `val` | Đánh giá OOD validation |
| `test` | Đánh giá cuối cùng theo benchmark |

Cần kiểm tra tài liệu của phiên bản WILDS đang sử dụng để xác nhận ý nghĩa chính xác của từng split.

Không tự chia ngẫu nhiên toàn bộ ảnh thành train và test, vì việc này có thể đưa dữ liệu từ cùng một bệnh viện vào cả hai tập và không đánh giá đúng mục tiêu domain generalization.

## 3. Cấu trúc thư mục

```text
camelyon17-project/
│
├── data/
│
├── results/
│
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── experiments.py
├── requirements.txt
└── README.md
```

### `data/`

Thư mục lưu dữ liệu Camelyon17-WILDS được tải và quản lý bởi thư viện WILDS.

Không nên tự chỉnh sửa hoặc trộn các split gốc của benchmark.

### `results/`

Thư mục lưu kết quả thực nghiệm, checkpoint và lịch sử huấn luyện.

Các tệp dự kiến gồm:

- `metrics.csv`: bảng tổng hợp kết quả của từng thí nghiệm.
- `erm_seed42_best.pth`: checkpoint tốt nhất của ERM.
- `erm_seed42_history.json`: lịch sử huấn luyện của ERM.
- Các tệp tương ứng cho `erm_aug`, `group_dro` và `group_dro_aug`.

Tên tệp có thể thay đổi tùy cấu hình chạy.

## 4. Mô tả chi tiết các file Python

### 4.1. `dataset.py` — Quản lý dữ liệu

File này chịu trách nhiệm tải dataset, tạo các split và xây dựng DataLoader.

Các chức năng chính:

1. Tải Camelyon17-WILDS thông qua `get_dataset()`.
2. Tạo các split `train`, `id_val`, `val` và `test`.
3. Xây dựng pipeline tiền xử lý ảnh.
4. Tạo DataLoader để cung cấp dữ liệu theo batch.
5. Xác định vị trí metadata của bệnh viện để phục vụ Group DRO và đánh giá theo domain.

Hai chế độ tiền xử lý:

- Baseline: resize, chuyển ảnh thành tensor và chuẩn hóa.
- Augmentation: thêm các phép biến đổi như lật ảnh, xoay ảnh và thay đổi màu sắc trong quá trình training.

Validation và test sử dụng cùng pipeline đánh giá cố định để đảm bảo việc so sánh công bằng.

**Đầu ra:** các DataLoader tương ứng với từng split và metadata cần thiết cho quá trình huấn luyện.

### 4.2. `model.py` — Xây dựng mô hình

File này định nghĩa kiến trúc ResNet-18.

Mô hình sử dụng trọng số pretrained từ ImageNet theo mặc định, sau đó thay lớp fully connected cuối cùng để phù hợp với bài toán phân loại nhị phân.

Quy trình:

1. Khởi tạo ResNet-18.
2. Nạp trọng số pretrained.
3. Thay lớp phân loại cuối.
4. Trả về mô hình để đưa vào quá trình training.

ResNet-18 được chọn làm baseline vì kiến trúc tương đối gọn, dễ huấn luyện và phù hợp để tiến hành các thí nghiệm đối chứng.

**Đầu ra:** mô hình dự đoán hai lớp từ ảnh mô bệnh học.

### 4.3. `train.py` — Huấn luyện mô hình

File này thực hiện một epoch huấn luyện và hỗ trợ hai phương pháp tối ưu.

#### Empirical Risk Minimization (ERM)

ERM tối thiểu hóa loss trung bình trên các mẫu huấn luyện:

$
R_{\mathrm{ERM}}(\theta)
=
\frac{1}{N}\sum_{i=1}^{N}
\ell(f_\theta(x_i),y_i)
$

Trong đó:

- $x_i$: ảnh đầu vào.
- $y_i$: nhãn thực tế.
- $f_\theta$: mô hình với tham số $\theta$.
- $\ell$: hàm loss, sử dụng Cross Entropy.

ERM đóng vai trò baseline để đo ảnh hưởng của domain shift.

#### Group DRO

Group DRO hướng đến việc giảm loss trên các domain khó:

$
R_{\mathrm{DRO}}(\theta)
=
\max_{g\in\mathcal G}R_g(\theta)
$

Trong đó $R_g$ là loss trung bình của domain $g$.

Triển khai sử dụng trọng số domain được cập nhật dựa trên loss của các nhóm trong batch. Những domain có loss cao được tăng trọng số để ảnh hưởng nhiều hơn đến mục tiêu huấn luyện.

Đây là phiên bản triển khai thực nghiệm cần được kiểm tra và hiệu chỉnh trước khi dùng làm kết quả nghiên cứu chính thức. Đặc biệt, cần xử lý sự thiếu vắng domain trong batch và theo dõi độ ổn định của trọng số nhóm.

**Đầu ra:** mô hình sau mỗi epoch và giá trị training loss.

### 4.4. `evaluate.py` — Đánh giá mô hình

File này tính các chỉ số đánh giá trên validation hoặc test.

Các chỉ số chính:

**Accuracy**

Tỷ lệ mẫu được phân loại chính xác:

$
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
$

**AUROC**

Đánh giá khả năng phân biệt hai lớp trên nhiều ngưỡng dự đoán.

**Cross Entropy Loss**

Đo mức độ sai lệch giữa xác suất dự đoán và nhãn thực tế.

**Average Group Risk**

Tính loss trung bình của từng domain rồi lấy trung bình giữa các domain:

$
R_{\mathrm{avg}}=
\frac{1}{G}\sum_{g=1}^{G}R_g
$

**Worst-Domain Risk**

Loss cao nhất trong các domain được đánh giá:

$
R_{\mathrm{worst}}=\max_g R_g
$

Các chỉ số theo domain cho phép kiểm tra xem hiệu suất trung bình có che giấu một bệnh viện có kết quả đặc biệt kém hay không.

Lưu ý: đánh giá worst-domain trên OOD chỉ mang ý nghĩa mô tả kết quả. Không được sử dụng nhãn OOD test để điều chỉnh mô hình.

### 4.5. `experiments.py` — Điều phối các thí nghiệm

Đây là file chính để chạy project.

Các chức năng:

1. Nhận cấu hình từ command line.
2. Đặt random seed.
3. Tạo DataLoader.
4. Khởi tạo mô hình và optimizer.
5. Gọi hàm huấn luyện từ `train.py`.
6. Đánh giá ID validation và OOD validation sau mỗi epoch.
7. Lưu checkpoint dựa trên ID validation.
8. Nạp checkpoint đã chọn để đánh giá test cuối cùng.
9. Ghi kết quả vào CSV và JSON.

Bốn cấu hình thí nghiệm:

| Tên | Phương pháp | Augmentation |
|---|---|---|
| `erm` | ERM | Không |
| `erm_aug` | ERM | Có |
| `group_dro` | Group DRO | Không |
| `group_dro_aug` | Group DRO | Có |

Cần giữ cố định các điều kiện so sánh như kiến trúc, số epoch, batch size và random seeds để hạn chế các yếu tố gây nhiễu.

### 4.6. `requirements.txt` — Các thư viện

Danh sách thư viện cần cài đặt:

- `torch`: xây dựng và huấn luyện mô hình Deep Learning.
- `torchvision`: ResNet-18 và các phép tiền xử lý ảnh.
- `wilds`: tải và truy cập benchmark WILDS.
- `scikit-learn`: tính Accuracy và AUROC.
- `numpy`: xử lý dữ liệu số.
- `pandas`: hỗ trợ xử lý bảng kết quả nếu cần mở rộng.
- `matplotlib`: trực quan hóa kết quả.
- `tqdm`: hiển thị tiến trình xử lý nếu được sử dụng.

## 5. Cài đặt môi trường

Tạo môi trường ảo:

```bash
python -m venv .venv
```

Kích hoạt trên Windows:

```bash
.venv\Scripts\activate
```

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

Nếu cần GPU NVIDIA, hãy cài bản PyTorch tương thích với CUDA theo hướng dẫn chính thức:

https://pytorch.org/get-started/locally/

Kiểm tra thiết bị:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu kết quả là `True`, PyTorch có thể nhận GPU trong môi trường hiện tại.

## 6. Cách chạy project

### 6.1. Chạy baseline ERM

```bash
python experiments.py --epochs 5 --batch-size 16 --runs erm
```

### 6.2. Chạy ERM với Data Augmentation

```bash
python experiments.py --epochs 5 --batch-size 16 --runs erm_aug
```

### 6.3. Chạy Group DRO

```bash
python experiments.py --epochs 5 --batch-size 16 --runs group_dro
```

### 6.4. Chạy tất cả các thí nghiệm

```bash
python experiments.py --epochs 5 --batch-size 16 --runs erm erm_aug group_dro group_dro_aug
```

### 6.5. Chạy nhiều random seeds

```bash
python experiments.py --epochs 5 --batch-size 16 --seeds 42 123 2026 --runs erm erm_aug group_dro group_dro_aug
```

Lần chạy đầu tiên cần tải dữ liệu và có thể tải trọng số pretrained của ResNet-18. Hãy chuẩn bị đủ dung lượng ổ đĩa và thời gian xử lý.

Các cấu hình trên là cấu hình khởi đầu để kiểm tra pipeline, không phải mặc định tối ưu cho benchmark.


### 6.6. Sử dụng archive Camelyon17 đã có trên Google Drive (Google Colab)

Nếu đã có file `camelyon17_v1.0.tar.gz` trên Google Drive, hãy kiểm tra file trước khi chạy để tránh tải lại archive lớn.

Mount Drive trong một cell Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Kiểm tra đường dẫn và dung lượng:

```python
from pathlib import Path

archive = Path('/content/drive/MyDrive/camelyon17_v1.0.tar.gz')
print('Archive tồn tại:', archive.is_file())
if archive.is_file():
    print(f'Dung lượng: {archive.stat().st_size / (1024**3):.2f} GiB')
```

Chạy thí nghiệm bằng archive có sẵn:

```bash
python experiments.py --epochs 5 --batch-size 16 --runs erm --archive-path /content/drive/MyDrive/camelyon17_v1.0.tar.gz
```

Nếu không truyền `--archive-path`, loader cũng kiểm tra một số vị trí mặc định, bao gồm `/content/drive/MyDrive/camelyon17_v1.0.tar.gz`. Loader cố gắng tạo liên kết tới archive ở vị trí WILDS mong đợi để tránh sao chép file nhiều GB. Nếu môi trường không hỗ trợ symlink, hãy đặt archive thủ công vào `<data-dir>/camelyon17_v1.0/archive.tar.gz`.

**Lưu ý:** tìm thấy archive không có nghĩa dữ liệu đã được giải nén. WILDS vẫn cần đọc/giải nén dữ liệu và kiểm tra cấu trúc thư mục. Nếu gặp lỗi, hãy kiểm tra nội dung archive và dung lượng ổ đĩa trước khi cho phép tải lại.

### 6.7. Tiếp tục huấn luyện và xử lý checkpoint

Các file checkpoint được lưu trong thư mục `results/`:

- `*_latest.pt`: lưu trạng thái gần nhất, gồm trọng số mô hình, optimizer, epoch, lịch sử và trọng số nhóm nếu dùng Group DRO. Đây là loại checkpoint ưu tiên khi cần resume.
- `*_best.pth`: lưu trọng số mô hình tốt nhất theo ID validation. File này không lưu optimizer và toàn bộ trạng thái huấn luyện nên không thể khôi phục chính xác như `*_latest.pt`.

Ví dụ tiếp tục thí nghiệm ERM đến epoch 10:

```bash
python experiments.py --epochs 10 --batch-size 16 --runs erm --resume --archive-path /content/drive/MyDrive/camelyon17_v1.0.tar.gz
```

Chỉ resume khi cấu hình thí nghiệm, seed và mô hình khớp với checkpoint. Nếu checkpoint `latest` bị lỗi hoặc không đọc được, phiên bản code đã bổ sung xử lý dự phòng để thử checkpoint `best` nếu có thể; nếu không có checkpoint hợp lệ thì cần bắt đầu lại. Hãy sao lưu checkpoint lỗi trước khi thử khôi phục.

Để giảm nguy cơ file checkpoint bị ghi dở, chương trình ghi ra file tạm rồi mới thay thế file đích. Nếu đang dùng Google Drive và kết nối không ổn định, nên lưu checkpoint trên ổ đĩa local trước rồi sao chép file hoàn chỉnh sang Drive.


## 7. Phương pháp thực nghiệm

### 7.1. Q1 — Performance loss do domain shift

Sử dụng baseline ERM và so sánh hiệu suất ID với OOD:

$
Gap_{\mathrm{acc}}
=
Acc_{\mathrm{ID}}-Acc_{\mathrm{OOD}}
$

Đồng thời báo cáo suy giảm tương đối:

$
Loss_{\mathrm{relative}}
=
\frac{Acc_{\mathrm{ID}}-Acc_{\mathrm{OOD}}}
{Acc_{\mathrm{ID}}}\times100\%
$

Gap dương cho thấy Accuracy trên OOD thấp hơn ID. Tuy nhiên, gap không tự chứng minh domain shift là nguyên nhân duy nhất.

### 7.2. Q2 — Hiệu quả của Data Augmentation

So sánh `erm` và `erm_aug`.

Chỉ số quan trọng nhất là sự thay đổi OOD Accuracy:

$
\Delta_{\mathrm{OOD}}
=
Acc_{\mathrm{OOD,aug}}
-
Acc_{\mathrm{OOD,baseline}}
$

Nếu giá trị dương, augmentation cải thiện OOD Accuracy trong thí nghiệm đó.

Cần đồng thời báo cáo ID Accuracy, AUROC và độ biến thiên qua nhiều seed.

### 7.3. Q3 — Average Risk hay Worst-Domain Risk

So sánh `erm` với `group_dro`.

Các chỉ số chính:

- Average Group Risk.
- Worst-Domain Risk.
- ID Accuracy và OOD Accuracy.
- Hiệu suất của từng domain.

Group DRO có thể cải thiện domain khó nhưng không bảo đảm luôn đạt kết quả OOD tốt hơn. Đây là giả thuyết cần kiểm chứng bằng thực nghiệm.

## 8. Quy tắc đánh giá công bằng

1. Sử dụng các split chính thức của WILDS.
2. Không sử dụng test để lựa chọn siêu tham số.
3. Sử dụng cùng kiến trúc và ngân sách huấn luyện giữa các phương pháp.
4. Dùng cùng pipeline đánh giá cho tất cả mô hình.
5. Báo cáo nhiều random seeds và độ lệch chuẩn khi có đủ tài nguyên.
6. Kiểm tra metadata bệnh viện và cách ánh xạ group ID trước khi chạy Group DRO.
7. Ghi rõ cấu hình, phiên bản thư viện và cách lựa chọn checkpoint trong báo cáo.

## 9. Kết quả cần trình bày

Bảng tổng hợp nên có cấu trúc như sau:

| Model | ID Accuracy | OOD Accuracy | ID AUROC | OOD AUROC | Average Risk | Worst Risk |
|---|---:|---:|---:|---:|---:|---:|
| ERM | ... | ... | ... | ... | ... | ... |
| ERM + Augmentation | ... | ... | ... | ... | ... | ... |
| Group DRO | ... | ... | ... | ... | ... | ... |
| Group DRO + Augmentation | ... | ... | ... | ... | ... | ... |

Các ô được điền bằng kết quả thực tế sau khi chạy chương trình.

Nên tạo thêm ba biểu đồ:

1. So sánh ID và OOD performance giữa các mô hình.
2. Training loss và validation loss theo epoch.
3. Loss hoặc Accuracy của từng bệnh viện.

## 10. Giới hạn của dự án

- Kết quả phụ thuộc vào cách chia domain, random seed và cấu hình huấn luyện.
- Augmentation có thể không phản ánh đầy đủ sự khác biệt thực tế giữa các bệnh viện.
- Group DRO cần xử lý đúng metadata và được điều chỉnh siêu tham số.
- ResNet-18 chỉ là baseline; kết quả chưa đại diện cho toàn bộ các kiến trúc Deep Learning.
- Kết quả trên Camelyon17 không thể tự động khái quát sang các bài toán y tế khác.

Đây là project nghiên cứu học thuật về độ bền vững và khả năng tổng quát hóa của mô hình, không phải hệ thống chẩn đoán lâm sàng.


## 11. Quy tắc quản lý dữ liệu và mã nguồn

Không đưa dataset Camelyon17, ảnh mô bệnh học, checkpoint lớn, log hoặc thông tin nhạy cảm lên GitHub. Chỉ lưu mã nguồn, cấu hình và tài liệu hướng dẫn trong repository. Dataset cần được quản lý riêng theo điều khoản sử dụng của nguồn dữ liệu.

File `.gitignore` trong repository được cấu hình để loại trừ dữ liệu và checkpoint phổ biến. Trước khi commit, hãy kiểm tra danh sách file được Git theo dõi, đặc biệt nếu dataset hoặc checkpoint từng được thêm vào repo trước đó.

## 12. Tài liệu tham khảo

1. Koh et al. (2021), *WILDS: A Benchmark of in-the-Wild Distribution Shifts*. Proceedings of ICML.
   https://proceedings.mlr.press/v139/koh21a.html

2. WILDS official website.
   https://wilds.stanford.edu/

3. WILDS source code.
   https://github.com/p-lambda/wilds

4. PyTorch documentation.
   https://pytorch.org/docs/stable/index.html

5. Torchvision documentation.
   https://pytorch.org/vision/stable/index.html
