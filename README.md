# Deep Learning Under Distribution Shift: WILDS Camelyon17

## 1. Giới thiệu dự án

### 1.1. Mục tiêu
Dự án khảo sát ảnh hưởng của domain shift đến khả năng tổng quát hóa của mô hình Deep Learning trên benchmark Camelyon17-WILDS. Dữ liệu gồm ảnh mô bệnh học được thu thập từ nhiều bệnh viện; mỗi bệnh viện được xem là một domain.

Mô hình được huấn luyện trên các hospital thuộc tập training và đánh giá trên hospital khác để đo mức suy giảm hiệu suất khi phân phối dữ liệu thay đổi.

### 1.2. Câu hỏi nghiên cứu
- **Q1 – Domain shift:** Hiệu suất giảm bao nhiêu khi chuyển từ In-Distribution (ID) sang Out-of-Distribution (OOD)?
- **Q2 – Data augmentation:** Augmentation có cải thiện khả năng tổng quát hóa OOD so với baseline không augmentation không?
- **Q3 – Average risk hay worst-domain risk:** ERM và GroupDRO khác nhau thế nào khi xét loss trung bình giữa các domain và loss của domain tệ nhất?

### 1.3. Thiết kế thực nghiệm
Dự án sử dụng ResNet-18 pretrained ImageNet, Cross Entropy và Adam. Bốn cấu hình:

| Cấu hình | Phương pháp | Augmentation |
|---|---|---|
| erm | ERM | Không |
| erm_aug | ERM | Có |
| group_dro | GroupDRO | Không |
| group_dro_aug | GroupDRO | Có |

Cấu hình thực nghiệm chính trong hướng dẫn: **15 epochs, batch size 16, seeds 42, 123, 2026**. Chạy ít epoch chỉ để kiểm tra pipeline, không thay thế kết quả thực nghiệm chính.

## 2. Dataset

### 2.1. Camelyon17-WILDS
Camelyon17-WILDS gồm ảnh RGB mô bệnh học hạch bạch huyết, dùng để phân loại vùng mô có chứa mô khối u hay không.

- Input: ảnh RGB 96 × 96.
- Target: nhãn nhị phân.
- Domain: bệnh viện thu thập ảnh.
- Metadata: chứa hospital ID, dùng cho GroupDRO và đánh giá theo domain.

Khác biệt giữa bệnh viện có thể đến từ quy trình xử lý mẫu, nhuộm màu và thu nhận ảnh; đây là nguồn domain shift cần khảo sát.

Nguồn tham khảo:
- Dataset: https://wilds.stanford.edu/datasets/
- WILDS source code: https://github.com/p-lambda/wilds
- Paper: https://proceedings.mlr.press/v139/koh21a.html

### 2.2. Các split được sử dụng

| Split trong code | Hospital | Vai trò |
|---|---|---|
| train | H0, H3, H4 | Huấn luyện |
| id_val | H0, H3, H4 | Validation ID và chọn checkpoint |
| val | H1 | Validation OOD |
| test | H2 | Held-out test, chỉ báo cáo sau khi chốt mô hình |

Q3 chính được tính trên bốn hospital validation H0, H3, H4, H1. H2 thuộc test được báo cáo riêng; không gộp H2 vào Q3 validation, không dùng H2 để chọn checkpoint hoặc tuning.

> Mapping split/hospital trên là cấu hình Camelyon17-WILDS mà code hiện tại kỳ vọng. Nếu thay phiên bản WILDS hoặc dữ liệu, cần kiểm tra lại metadata và thống kê domain.

## 3. Cấu trúc thư mục

    hoc-sau-camelyon17/
    ├── data/                 # Dataset WILDS, không commit lên Git
    ├── results/              # Kết quả local, không commit checkpoint/log
    ├── dataset.py            # Dataset, transforms, DataLoader, domain stats
    ├── model.py              # ResNet-18
    ├── train.py              # Một epoch ERM và GroupDRO
    ├── evaluate.py           # Metrics tổng thể và theo hospital
    ├── experiments.py        # Điều phối train/evaluate, checkpoint, CSV
    ├── analysis.py            # Phân tích Q1-Q3 và tạo biểu đồ
    ├── requirements.txt      # Thư viện Python
    ├── .gitignore
    └── README.md

### Thư mục kết quả
Mỗi lần chạy cần chỉ định rõ results directory. Trên Google Colab của dự án, thư mục kết quả thống nhất là:

    /content/drive/MyDrive/hoc-sau-results

Các tệp chính do experiments.py tạo:
- metrics.csv: một dòng cho mỗi model/seed, gồm ID/OOD/test metrics và risk theo validation/hospital.
- domain_stats.csv: số mẫu và tỷ lệ tumor theo split/hospital.
- {model}_seed{seed}_best.pth: trọng số tốt nhất theo ID validation.
- {model}_seed{seed}_latest.pt: trạng thái resume gồm model, optimizer, epoch, history, run config và group weights (nếu dùng GroupDRO).
- {model}_seed{seed}_history.json: train loss, ID validation và OOD validation theo epoch.

Các biểu đồ do analysis.py tạo:
- q1_id_ood_accuracy.png: ID validation so với OOD validation accuracy.
- q1_loss_curves.png: train loss, ID validation loss và OOD validation loss theo epoch.
- q3_validation_risk.png: average risk và worst-domain risk trên bốn hospital validation.
- q3_validation_hospital_accuracy.png: accuracy theo từng hospital validation.
- q3_all5_posthoc_accuracy.png: accuracy năm hospital, gồm H2 test; chỉ là biểu đồ hậu nghiệm.

## 4. Mô tả các file Python

### 4.1. dataset.py – Quản lý dữ liệu
- Gọi wilds.get_dataset(dataset="camelyon17") để truy cập benchmark.
- Kiểm tra archive có sẵn và ưu tiên dùng dữ liệu đã giải nén hoặc archive được chỉ định.
- Tạo bốn split: train, id_val, val, test.
- Xây dựng DataLoader, lấy chỉ số metadata hospital và xác định số nhóm.
- Cung cấp get_domain_stats() để thống kê số mẫu, số mẫu tumor/non-tumor và tỷ lệ tumor theo split/hospital.

Transform đánh giá gồm resize 96×96, ToTensor và chuẩn hóa ImageNet. Khi bật augmentation cho training, code thêm RandomHorizontalFlip, RandomVerticalFlip, RandomRotation và ColorJitter. Validation/test luôn dùng transform cố định, không augmentation.

### 4.2. model.py – Xây dựng mô hình
Định nghĩa ResNet-18 với trọng số pretrained ImageNet (ResNet18_Weights.DEFAULT), thay fully-connected layer cuối để dự đoán hai lớp. Lần đầu khởi tạo có thể cần Internet để tải pretrained weights nếu chúng chưa có trong cache.

### 4.3. train.py – Huấn luyện một epoch
**ERM:** tính Cross Entropy trung bình trên các mẫu trong minibatch và cập nhật mô hình bằng backpropagation.

**GroupDRO:** duy trì trọng số cho các hospital thuộc training; cập nhật trọng số theo multiplicative update dựa trên group loss. Loss minibatch được chuẩn hóa trên những nhóm xuất hiện trong batch.

**Giới hạn triển khai:** đây là stochastic minibatch approximation của GroupDRO, không phải phép tối ưu full-batch chính xác. Vì minibatch có thể chứa tập hospital khác nhau, kết quả cần được đánh giá qua nhiều seed và cần mô tả đúng giới hạn này trong báo cáo. Training loss được log là sample-average Cross Entropy để theo dõi, không phải GroupDRO objective.

### 4.4. evaluate.py – Đánh giá mô hình
Tính Accuracy, AUROC, Cross Entropy Loss, số mẫu và metrics theo từng hospital (risk và accuracy).

- evaluate_validation_domains() gộp H0/H3/H4 từ ID validation với H1 từ OOD validation để tính average risk, worst-domain risk, worst risk trên nhóm seen/unseen và worst accuracy.
- evaluate_all_hospitals() bổ sung H2 test cho báo cáo hậu nghiệm, không dùng chọn mô hình.

Average Group Risk là trung bình không trọng số của Cross Entropy risk từng hospital. Worst-Domain Risk là risk lớn nhất trong các hospital thuộc tập đang xét.

### 4.5. experiments.py – Điều phối thí nghiệm
Đây là entry point chính. File này:
1. Đọc tham số command line: epochs, batch size, learning rate, seeds, runs, data/results paths.
2. Thiết lập random seeds.
3. Tạo DataLoader baseline và DataLoader augmentation riêng khi cần.
4. Khởi tạo ResNet-18, Adam và group weights.
5. Gọi hàm train tương ứng trong train.py.
6. Đánh giá ID validation và OOD validation sau mỗi epoch.
7. Chọn best.pth theo **ID validation accuracy**; không chọn theo OOD test.
8. Lưu latest.pt để resume và history.json để phân tích.
9. Đánh giá checkpoint tốt nhất trên ID validation, OOD validation và test.
10. Tính Q3 validation-only, thống kê năm hospital hậu nghiệm và cập nhật metrics.csv.

Khi chạy lại cùng model/seed trong cùng results directory, dòng tương ứng trong metrics.csv được cập nhật và checkpoint cùng tên có thể bị ghi đè. Hãy sao lưu trước nếu cần giữ kết quả cũ.

### 4.6. analysis.py – Phân tích kết quả Q1–Q3
File này **không huấn luyện lại mô hình**. Nó đọc metrics.csv và các history JSON trong results directory để:
- **Q1:** tổng hợp ID/OOD Accuracy, AUROC, gap tuyệt đối, relative accuracy drop và loss change theo model; báo cáo mean ± standard deviation qua seeds.
- **Q2:** ghép cặp cùng seed giữa erm và erm_aug, cũng như group_dro và group_dro_aug; tính delta augmentation cho OOD Accuracy/AUROC/Loss và ID metrics.
- **Q3:** tổng hợp average risk, worst-domain risk, worst seen/unseen risk trên bốn hospital validation. Nếu metrics cũ thiếu cột validation-4, script cảnh báo và không coi all-five risk là Q3 chính.
- Tạo các biểu đồ PNG được liệt kê ở mục 3.

Chạy sau khi thí nghiệm hoàn tất: python analysis.py --results-dir <thư_mục_kết_quả>.

### 4.7. requirements.txt và .gitignore
requirements.txt liệt kê PyTorch, Torchvision, WILDS, scikit-learn, NumPy, Pandas, Matplotlib và tqdm. .gitignore giúp tránh commit dataset, checkpoint, log và tệp runtime lớn; vẫn cần kiểm tra git status trước khi push.

## 5. Cài đặt môi trường

### 5.1. Cài thư viện

    python -m venv .venv

Windows:
    .venv\Scripts\activate

Linux/macOS:
    source .venv/bin/activate

Cài thư viện:
    pip install -r requirements.txt

Nếu dùng GPU NVIDIA, cài PyTorch/Torchvision tương thích CUDA theo hướng dẫn: https://pytorch.org/get-started/locally/

Kiểm tra GPU:
    python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

### 5.2. Chuẩn bị dataset
Có thể để WILDS tải dataset vào data/ trong lần chạy đầu (cần Internet và đủ dung lượng), hoặc dùng archive đã tải sẵn. Trên Colab, archive là:

    /content/drive/MyDrive/camelyon17_v1.0.tar.gz

Nếu dữ liệu đã giải nén, thư mục được code kỳ vọng là:

    /content/hoc-sau-camelyon17/data/camelyon17_v1.0/

Bên trong cần có metadata.csv và thư mục patches/. Nếu đã giải nén đầy đủ, không cần truyền --archive-path.

## 6. Cách chạy project

### 6.1. Chạy thử để kiểm tra pipeline
Ví dụ chạy ERM một seed trong 1 epoch để kiểm tra môi trường (không phải kết quả chính thức):

    python experiments.py --epochs 1 --batch-size 16 --seeds 42 --runs erm --data-dir data --results-dir results/smoke_test

### 6.2. Cấu hình thực nghiệm chính: 15 epochs, 3 seeds
Các lệnh dưới đây chạy đến tổng cộng 15 epochs, batch size 16. Chạy toàn bộ bốn cấu hình:

    python experiments.py --epochs 15 --batch-size 16 --seeds 42 123 2026 --runs erm erm_aug group_dro group_dro_aug --data-dir data --results-dir results/main

Có thể chạy riêng từng cấu hình bằng cách thay giá trị --runs, ví dụ:

    python experiments.py --epochs 15 --batch-size 16 --seeds 42 123 2026 --runs group_dro --data-dir data --results-dir results/main

Các cấu hình giữ cùng kiến trúc, batch size, learning rate mặc định 1e-4, epoch và seed để hỗ trợ so sánh paired. Kết quả vẫn có thể dao động theo seed; cần báo cáo mean và standard deviation.

### 6.3. Chạy trên Google Colab với dữ liệu và kết quả trên Drive

Mount Drive:

    from google.colab import drive
    drive.mount('/content/drive')

Thiết lập và kiểm tra đường dẫn:

    from pathlib import Path

    PROJECT = Path('/content/hoc-sau-camelyon17')
    DATA = PROJECT / 'data'
    RESULTS = Path('/content/drive/MyDrive/hoc-sau-results')
    RESULTS.mkdir(parents=True, exist_ok=True)

    assert (DATA / 'camelyon17_v1.0' / 'metadata.csv').exists()
    assert (DATA / 'camelyon17_v1.0' / 'patches').exists()
    print('Project:', PROJECT)
    print('Data:', DATA)
    print('Results:', RESULTS)

Nếu dữ liệu đã giải nén tại vị trí trên, chạy toàn bộ thí nghiệm chính:

    %cd /content/hoc-sau-camelyon17
    !python experiments.py --epochs 15 --batch-size 16 --seeds 42 123 2026 --runs erm erm_aug group_dro group_dro_aug --data-dir /content/hoc-sau-camelyon17/data --results-dir /content/drive/MyDrive/hoc-sau-results

Nếu chỉ muốn chạy GroupDRO:

    !python experiments.py --epochs 15 --batch-size 16 --seeds 42 123 2026 --runs group_dro --data-dir /content/hoc-sau-camelyon17/data --results-dir /content/drive/MyDrive/hoc-sau-results

Nếu chưa giải nén nhưng có archive trên Drive, truyền đường dẫn archive:

    !python experiments.py --epochs 15 --batch-size 16 --seeds 42 123 2026 --runs erm erm_aug group_dro group_dro_aug --data-dir /content/hoc-sau-camelyon17/data --archive-path /content/drive/MyDrive/camelyon17_v1.0.tar.gz --results-dir /content/drive/MyDrive/hoc-sau-results

Loader ưu tiên dữ liệu đã giải nén nếu thấy metadata.csv và patches/. Nếu chỉ thấy archive, WILDS có thể cần giải nén/tổ chức dữ liệu; hãy kiểm tra dung lượng và cấu trúc trước khi để thư viện tải lại dữ liệu.

### 6.4. Resume khi phiên Colab bị ngắt
--epochs 15 là **tổng số epoch mục tiêu**, không phải số epoch cộng thêm. Ví dụ checkpoint đã lưu epoch 7 thì chạy lại cùng lệnh với --resume để tiếp tục từ epoch 8 đến epoch 15:

    %cd /content/hoc-sau-camelyon17
    !python experiments.py --epochs 15 --batch-size 16 --seeds 2026 --runs group_dro --resume --data-dir /content/hoc-sau-camelyon17/data --results-dir /content/drive/MyDrive/hoc-sau-results

Chỉ resume checkpoint có run_config tương thích với method, augmentation, batch size, learning rate và training implementation version hiện tại. Checkpoint legacy không có cấu hình tương thích sẽ bị từ chối có chủ đích. Không thể resume chính xác từ best.pth vì file này chỉ chứa model weights; cần latest.pt để khôi phục optimizer, epoch và group weights.

Nếu latest checkpoint bị hỏng, không xóa ngay. Sao lưu và kiểm tra trước; không đổi tên best.pth thành latest.pt vì hai file lưu trạng thái khác nhau.

### 6.5. Bảo toàn checkpoint cũ trước khi chạy lại
Các lần chạy cùng model/seed trong cùng results directory có thể ghi đè checkpoint và cập nhật dòng tương ứng trong metrics.csv. Trước khi chạy lại cấu hình bị ảnh hưởng bởi lỗi augmentation/GroupDRO cũ, sao lưu các tệp trong chính thư mục Drive:

    from pathlib import Path
    import shutil

    root = Path('/content/drive/MyDrive/hoc-sau-results')
    patterns = [
        'erm_aug_seed*_best.pth', 'erm_aug_seed*_latest.pt', 'erm_aug_seed*_history.json',
        'group_dro_seed*_best.pth', 'group_dro_seed*_latest.pt', 'group_dro_seed*_history.json',
        'group_dro_aug_seed*_best.pth', 'group_dro_aug_seed*_latest.pt', 'group_dro_aug_seed*_history.json',
    ]
    for pattern in patterns:
        for src in root.glob(pattern):
            dst = src.with_name(src.name + '.legacy')
            if not dst.exists():
                shutil.copy2(src, dst)
            print(src.name, '->', dst.name)

Sau khi backup, huấn luyện lại các cấu hình bị ảnh hưởng bằng code hiện tại, không dùng --resume:

    !python experiments.py --epochs 15 --batch-size 16 --seeds 42 123 2026 --runs erm_aug group_dro group_dro_aug --data-dir /content/hoc-sau-camelyon17/data --results-dir /content/drive/MyDrive/hoc-sau-results

ERM baseline cũ có thể giữ nếu được tạo từ cùng kiến trúc, baseline transform và cấu hình tương ứng. Không dùng kết quả augmentation cũ làm bằng chứng Q2 nếu augmentation chưa thực sự được áp dụng.

### 6.6. Đánh giá lại checkpoint và phân tích Q1–Q3
Chỉ đánh giá lại checkpoint tốt nhất đã có, không huấn luyện:

    python experiments.py --evaluate-only --checkpoint-dir /path/to/old_results --results-dir results/re_evaluation --runs erm erm_aug group_dro group_dro_aug --seeds 42 --data-dir data

--evaluate-only đọc best.pth từ checkpoint directory và đánh giá ID validation, OOD validation, test. Nó không kiểm chứng lại cách mô hình đã được huấn luyện; đánh giá lại checkpoint legacy không thể sửa lỗi augmentation hoặc thuật toán trong quá trình train. Dùng results directory riêng để không trộn kết quả cũ với kết quả corrected. Nếu model/seed trùng nhau, metrics tương ứng trong file đích sẽ được cập nhật.

Sau khi thí nghiệm hoàn tất, phân tích kết quả:

    python analysis.py --results-dir results/main

Trên Colab:

    !python analysis.py --results-dir /content/drive/MyDrive/hoc-sau-results

Các bảng và biểu đồ được ghi vào cùng results directory; script không train lại mô hình.

## 7. Phương pháp phân tích Q1–Q3

### 7.1. Q1 – Performance loss do domain shift
So sánh ID validation (H0, H3, H4) với OOD validation (H1):

- Accuracy gap = ID Accuracy - OOD Accuracy.
- Relative accuracy drop = (ID Accuracy - OOD Accuracy) / ID Accuracy × 100%.

Báo cáo thêm AUROC gap và thay đổi Cross Entropy Loss. Gap dương cho thấy metric giảm trên OOD validation; không nên khẳng định domain shift là nguyên nhân duy nhất nếu chưa loại trừ yếu tố khác.

### 7.2. Q2 – Hiệu quả của augmentation
So sánh cùng seed giữa erm và erm_aug, đồng thời giữa group_dro và group_dro_aug.

OOD delta = OOD metric (augmentation) - OOD metric (baseline). Với Accuracy/AUROC, delta dương thường tốt hơn; với Loss, delta âm thường tốt hơn. Phải xác nhận augmentation được bật ở training và validation/test vẫn giữ transform cố định.

### 7.3. Q3 – Average Risk và Worst-Domain Risk
So sánh ERM với GroupDRO bằng:
- Average risk validation: trung bình risk của H0, H3, H4, H1.
- Worst-domain risk validation: risk lớn nhất trong bốn hospital validation.
- Worst seen risk: lớn nhất trong H0, H3, H4.
- Worst unseen validation risk: risk của H1.
- Accuracy/risk riêng từng hospital.

H2 test được báo cáo riêng sau khi chốt mô hình. avg_risk_all5_posthoc và worst_risk_all5_posthoc chỉ là thống kê hậu nghiệm trên cả năm hospital, không phải metric Q3 chính để chọn mô hình.

## 8. Quy tắc đánh giá công bằng
1. Dùng split chính thức của WILDS và giữ nguyên metadata hospital.
2. Không dùng H2 test để chọn checkpoint hoặc tuning.
3. Giữ cùng kiến trúc, optimizer, learning rate, batch size và epoch giữa các cấu hình, trừ yếu tố đang nghiên cứu.
4. Augmentation chỉ áp dụng ở training; validation/test dùng transform cố định.
5. So sánh paired theo cùng random seeds và báo cáo mean ± standard deviation.
6. Ghi rõ GroupDRO hiện là stochastic minibatch approximation.
7. Lưu cấu hình, phiên bản thư viện và kết quả để có thể tái lập.

## 9. Kết quả cần trình bày

| Model | ID Acc | OOD Acc | ID AUROC | OOD AUROC | Validation Avg Risk | Validation Worst Risk |
|---|---:|---:|---:|---:|---:|---:|
| ERM | ... | ... | ... | ... | ... | ... |
| ERM + Augmentation | ... | ... | ... | ... | ... | ... |
| GroupDRO | ... | ... | ... | ... | ... | ... |
| GroupDRO + Augmentation | ... | ... | ... | ... | ... | ... |

Điền bảng bằng kết quả thực tế từ metrics.csv, không ghi số liệu minh họa như kết quả thực nghiệm. Khi tổng hợp nhiều seed, báo cáo mean ± standard deviation và số paired seeds cho Q2.

## 10. Giới hạn của dự án
- Camelyon17 chỉ đại diện cho một benchmark, không bao quát mọi domain shift trong dữ liệu y tế.
- Augmentation có thể không mô phỏng đầy đủ khác biệt giữa bệnh viện.
- GroupDRO trong code là stochastic minibatch approximation, không phải exact full-batch implementation.
- Kết quả phụ thuộc seed, số epoch, hyperparameter và cách chọn checkpoint.
- ResNet-18 là baseline, chưa đại diện cho mọi kiến trúc Deep Learning.
- Đây là nghiên cứu học thuật, không phải hệ thống chẩn đoán lâm sàng.

## 11. Quản lý dữ liệu và mã nguồn
Không commit dataset, ảnh mô bệnh học, checkpoint lớn hoặc log runtime lên GitHub. Chỉ lưu mã nguồn và tài liệu. Kiểm tra git status và git ls-files trước khi commit, đặc biệt nếu dữ liệu/checkpoint từng được thêm vào repository.

## 12. Tài liệu tham khảo
1. Koh et al. (2021), *WILDS: A Benchmark of in-the-Wild Distribution Shifts*. ICML. https://proceedings.mlr.press/v139/koh21a.html
2. WILDS official website: https://wilds.stanford.edu/
3. WILDS source code: https://github.com/p-lambda/wilds
4. PyTorch documentation: https://pytorch.org/docs/stable/index.html
5. Torchvision documentation: https://pytorch.org/vision/stable/index.html
