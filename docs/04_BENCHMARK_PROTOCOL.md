# Tài Liệu Kỹ Thuật: Giao Thức Đo Kiểm Chuẩn Hóa (Benchmark Protocol)
### Dự án Meetly - Speech-to-Text Research Benchmark

---

## 1. Mục Đích & Nguyên Tắc So Sánh Công Bằng (Fair Comparison)

Để kết quả nghiên cứu có giá trị thực tiễn và giúp đưa ra quyết định kiến trúc chính xác, toàn bộ quá trình đo kiểm phải tuân theo một **Giao thức thực nghiệm nghiêm ngặt (Controlled Benchmark Protocol)**:
- **Cùng tập dữ liệu**: Mọi mô hình được đánh giá trên cùng tập âm thanh và ground truth tương ứng.
- **Cùng phần cứng & môi trường**: Ghi nhận đầy đủ thông số qua `results/experiment_environment.json`.
- **Cùng chuẩn tiền xử lý**: Âm thanh đều đưa về 16kHz Mono float32 trước khi đưa vào mô hình.
- **Cùng chuẩn đánh giá văn bản**: Không phạt oan mô hình vì định dạng viết hoa, dấu câu hay lỗi mã hóa Unicode.
- **Không tính thời gian tải mô hình**: Thời gian tải weights từ HuggingFace Hub hoặc đọc từ ổ cứng không được tính vào thời gian suy luận (ASR inference time).

---

## 2. Chuẩn Hóa Dữ Liệu Đầu Vào & Đầu Ra

### A. Chuẩn Hóa Âm Thanh
Mọi file âm thanh từ micro, livestream hay video trích xuất đều được xử lý qua `common/audio.py`:
- **Tần số lấy mẫu (Sample Rate)**: Cố định chuẩn $16,000$ Hz (16 kHz). Nếu đầu vào có sample rate khác (44.1kHz, 48kHz), thực hiện resample chất lượng cao qua bộ lọc đa pha `scipy.signal.resample_poly`.
- **Số kênh (Channels)**: Chuyển toàn bộ về kênh đơn (Mono) bằng cách tính trung bình cộng giữa các kênh.
- **Biên độ (Amplitude Range)**: Chuẩn hóa về mảng numpy float32 trong khoảng $[-1.0, 1.0]$.

### B. Chuẩn Hóa Văn Bản Tiếng Việt 2 Lớp (`common/text_normalization.py`)
1. **Lớp 1: Normalized WER / CER (Tiêu chuẩn ASR)**
   - Chuẩn hóa Unicode về dạng **NFC** chuẩn (`unicodedata.normalize('NFC', text)`).
   - Đồng nhất vị trí dấu thanh tiếng Việt giữa cách gõ cũ và mới (ví dụ: `hoà` $\rightarrow$ `hòa`, `thuỷ` $\rightarrow$ `thủy`).
   - Chuyển toàn bộ ký tự sang chữ thường (lowercase).
   - Loại bỏ tất cả dấu câu và ký tự định dạng (`.,?!:;"'()[]{}-...`).
   - Gộp nhiều khoảng trắng liền kề thành 1 khoảng trắng đơn.
2. **Lớp 2: Phân Tích Lỗi Thuật Ngữ Công Nghệ (VI-EN Code-Switching)**
   - Đối với tập `codeswitch_tech`, trích xuất các thuật ngữ công nghệ tiếng Anh (*API, JWT, Spring Boot, Docker, backend, frontend, microservice...*).
   - Đo lường định lượng:
     $$\text{Technical Term Recall} = \frac{\text{Số thuật ngữ nhận diện đúng}}{\text{Tổng số thuật ngữ trong Ground Truth}}$$
     $$\text{Technical Term Accuracy} = \frac{\text{Số thuật ngữ nhận diện đúng}}{\text{Đúng} + \text{Thay thế} + \text{Bỏ sót}}$$

---

## 3. Quy Trình Thực Thi Đo Kiểm (Execution Protocol)

### A. Khởi Động (Warm-up Run)
- Trước khi bắt đầu bấm giờ đo chính thức, mỗi mô hình phải chạy qua **1 lượt warm-up** trên dữ liệu mẫu ngắn.
- Lượt warm-up giúp khởi tạo bộ nhớ GPU, nạp các thư viện CUDA JIT và làm nóng cache, tránh việc lượt chạy đầu tiên bị méo mó độ trễ do chi phí khởi tạo.

### B. Lặp Lại Đo Đạc (Repeated Runs)
- Mỗi tác vụ được thực thi lặp lại **3 lần chính thức** ($N = 3$).
- Thời gian suy luận được tính bằng trung bình cộng (Mean) của các lần chạy để triệt tiêu dao động ngẫu nhiên của hệ điều hành.

### C. Quy Tắc Xử Lý CUDA Cache
- **Tuyệt đối cấm**: Không gọi `torch.cuda.empty_cache()` giữa các bước incremental streaming hoặc giữa các đoạn phân đoạn nhỏ, vì thao tác đồng bộ GPU này gây trễ nhân tạo nghiêm trọng và làm sai lệch chỉ số P95/TTFP.
- **Chỉ cho phép**: Gọi `torch.cuda.empty_cache()` và `gc.collect()` **giữa các đợt benchmark run độc lập** để trả lại môi trường sạch sẽ cho mô hình tiếp theo.

---

## 4. Phương Pháp Đo Lường Tài Nguyên Phần Cứng 3 Chiều

Để tránh việc môi trường Colab hoặc các tiến trình khác chiếm sẵn bộ nhớ làm sai lệch kết quả:
- **Baseline Memory**: Đo dung lượng bộ nhớ đang chiếm dụng ngay trước khi mô hình chạy.
- **Peak Memory**: Dung lượng bộ nhớ cao nhất được ghi nhận trong quá trình thực thi.
- **Delta Peak Memory**: $\text{Delta} = \text{Peak} - \text{Baseline}$ (đây là chi phí thực sự mà mô hình tiêu tốn).
- **Nguồn đo VRAM**: Ưu tiên truy vấn trực tiếp qua **NVML** (`pynvml`) để hỗ trợ đồng thời và chính xác cho cả CTranslate2 và PyTorch allocator.

---

## 5. Hướng Dẫn Tái Lập Thí Nghiệm (Reproduction Steps)

### Chạy trên máy cục bộ (CPU Test & Verification)
```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Khởi tạo dữ liệu mẫu và chạy unit test
python data/prepare_dataset.py
python -m pytest tests

# 3. Ghi nhận vân tay môi trường
python -c "from common.result_schema import EnvironmentFingerprint; env = EnvironmentFingerprint.capture(); env.save_to_json('results/experiment_environment.json')"
```

### Chạy trên Google Colab / Server GPU (Full Benchmark)
1. Tải toàn bộ thư mục `meetly-stt-research/` lên Google Drive hoặc clone qua Git.
2. Chọn runtime **GPU (T4, L4 hoặc A100)**.
3. Cài đặt các thư viện phụ thuộc:
   ```bash
   !pip install -r requirements.txt
   !pip install faster-whisper ctranslate2 pynvml
   ```
4. Mở và chạy tuần tự các notebook:
   - `notebooks/01_model_screening.ipynb`
   - `notebooks/02_offline_pipeline_benchmark.ipynb`
   - `notebooks/03_streaming_pipeline_benchmark.ipynb`
5. Kết quả tự động lưu trữ vào `results/*.jsonl`, `results/*.csv` và các biểu đồ Pareto tại `results/figures/`.
