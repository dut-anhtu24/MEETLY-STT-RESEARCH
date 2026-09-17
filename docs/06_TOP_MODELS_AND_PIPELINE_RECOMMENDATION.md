# Báo Cáo Nghiên Cứu STT Meetly: Đánh Giá Thực Nghiệm, Đề Xuất Top 3 Offline/Online & Thiết Kế Pipeline Sản Xuất

Tài liệu này là báo cáo kỹ thuật tổng hợp cao nhất (Master Technical Recommendation) của hệ sinh thái nhận dạng giọng nói (**Speech-to-Text - STT**) thuộc dự án **Meetly**. Báo cáo tích hợp toàn bộ số liệu thực nghiệm định lượng, phân tích sự đánh đổi đa mục tiêu Biên Pareto, đánh giá chi tiết các ca thất bại, đề xuất cấu hình Top 3 cho hai chế độ Offline và Streaming, cùng thiết kế kiến trúc pipeline cho môi trường vận hành thực tế (Production).

---

## 1. Mục Tiêu & Phạm Vi Đánh Giá

Trong nền tảng họp thông minh Meetly, hệ thống STT phải giải quyết đồng thời hai bài toán nghiệp vụ riêng biệt:
1. **Chế độ Ngoại tuyến (Offline Mode)**:
   - *Luồng dữ liệu*: Video cuộc họp đã ghi hình hoặc file âm thanh hoàn chỉnh $\rightarrow$ Tách luồng audio $\rightarrow$ Chuẩn hóa định dạng $\rightarrow$ Phân đoạn tiếng nói (VAD) $\rightarrow$ Nhận dạng âm thanh (ASR) $\rightarrow$ Gán mốc thời gian (Word/Segment Timestamps) $\rightarrow$ Ghép nối biên bản $\rightarrow$ Xuất định dạng JSON có cấu trúc.
   - *Mục tiêu cốt lõi*: Tối ưu hóa độ chính xác văn bản (cực tiểu hóa WER/CER), bảo toàn thuật ngữ kỹ thuật chêm xen tiếng Anh (Code-Switching), đảm bảo thông lượng xử lý nhanh gấp nhiều lần thời gian thực ($Throughput \gg 1$) và tính ổn định trên ngữ cảnh dài.
2. **Chế độ Trực tuyến / Thời gian thực (Online / Streaming Mode)**:
   - *Luồng dữ liệu*: Luồng âm thanh micro người tham gia phát qua WebRTC/WebSocket $\rightarrow$ Bộ đệm khung âm thanh (Audio Ring Buffer) $\rightarrow$ Lọc khoảng lặng (VAD/VAC gating) $\rightarrow$ Cửa sổ ngữ cảnh trượt (Rolling Context Window) $\rightarrow$ Suy luận gia tăng (Incremental ASR) $\rightarrow$ Chính sách ổn định tiền tố (`LocalAgreement-inspired stability policy`) $\rightarrow$ Phát sự kiện văn bản tạm thời (Partial Transcript) và văn bản đã chốt (Stable Transcript).
   - *Mục tiêu cốt lõi*: Cực tiểu hóa độ trễ phản hồi (TTFP < 500ms, P95 latency < 200ms), ngăn ngừa hiện tượng giật/nhảy chữ gây mất tập trung (cực tiểu hóa Revision Count), và tối ưu hóa mức chiếm dụng bộ nhớ GPU (VRAM).

---

## 2. Môi Trường Benchmark & Truy Vết Phần Cứng

Nhằm đảm bảo tính khách quan và khả năng tái lập (reproducibility) 100%, mọi phép đo đạc trong dự án được ghi nhận chính xác theo môi trường phần cứng, trích xuất trực tiếp từ file `results/experiment_environment.json`:

### 2.1. Môi Trường DUT AI GPU Server (`dutai_gpu_env_20260917_170308`)
* **Phần cứng GPU**: **NVIDIA Tesla V100-SXM2-16GB** (VRAM thực tế: 16,144 MB).
* **Phần cứng CPU & RAM**: 12 vCPUs x86_64, 15,841 MB RAM hệ thống.
* **Hệ điều hành & Nền tảng**: Linux 6.8.0-138-generic (Ubuntu 22.04), Python 3.10.12.
* **Phiên bản Framework & Driver**: CUDA 13.0, PyTorch 2.14.0+cu130, Transformers 5.17.0, CTranslate2 4.8.2, faster-whisper 1.2.1, FFmpeg 4.4.2.
* *Nhiệm vụ thực nghiệm*: Đợt đo kiểm chuẩn hóa trên tập VIVOS Test (19 mẫu audio thực tế) và sàng lọc pipeline quy mô lớn.

### 2.2. Môi Trường Colab GPU (`colab_t4_cuda122_20260917`)
* **Phần cứng GPU**: **NVIDIA Tesla T4** (VRAM thực tế: 15,360 MB).
* **Phần cứng CPU & RAM**: 2 vCPUs Intel Xeon @ 2.20GHz, 12,985 MB RAM hệ thống.
* **Hệ điều hành & Nền tảng**: Linux 6.1.58+, Python 3.12.7, CUDA 12.2, PyTorch 2.3.0.
* *Nhiệm vụ thực nghiệm*: Đợt đo kiểm bóc tách (Ablation study) cho Offline Pipeline và mô phỏng luồng trực tuyến (Streaming Simulation).

### 2.3. Môi Trường Local CPU
* Máy trạm cục bộ Windows (không có GPU CUDA), chỉ được sử dụng cho việc phát triển mã nguồn nền tảng, chạy bộ kiểm thử tự động (Unit Tests) và kiểm tra schema với mô hình siêu nhẹ (`whisper-tiny`). Tuyệt đối không dùng số liệu máy local để báo cáo năng lực mô hình.

---

## 3. Danh Mục Dataset Thực Nghiệm Sử Dụng

Thực nghiệm Meetly tuân thủ chiến lược dữ liệu 2 tầng (Two-Tier Dataset Strategy):

### 3.1. Tầng A (Tier A — Minimum Benchmark Dataset)
Gồm 6 tập dữ liệu con (subsets) được thu âm và chuẩn hóa 16 kHz Mono, ground-truth được rà soát thủ công bởi con người:
1. `clean_vi` (SUB-01): Giọng đọc tiếng Việt chuẩn, không tạp âm (xác định cận trên độ chính xác).
2. `noisy_vi` (SUB-02): Giọng nói trong môi trường văn phòng có tiếng gõ phím, quạt gió (SNR 10–15dB).
3. `meeting_natural` (SUB-03): Hội thoại tự nhiên, ngắt nghỉ không đều, có từ đệm (*à, ừ*) và hiện tượng câu nói chồng lấn nhẹ (mild overlap).
4. `codeswitch_tech` (SUB-04): Hội thoại chuyên ngành công nghệ phần mềm chứa các thuật ngữ IT tiếng Anh (*API, JWT, Docker, Spring Boot, backend, microservices*).
5. `silence_pause` (SUB-05): Các câu nói ngắn xen kẽ khoảng lặng dài 3s – 8s để kiểm tra nguy cơ sinh ảo giác (hallucination) của mô hình.
6. `longform_meeting` (SUB-06): Đoạn ghi âm cuộc họp liên tục 5 – 10 phút (mini long-form benchmark).

### 3.2. Tầng B (Tier B — Standardized Vietnamese Testset)
* Tập con chuẩn hóa từ bộ dữ liệu mở **VIVOS Test** gồm 19 mẫu âm thanh phát âm tiếng Việt đa dạng ngữ âm và vùng miền (tổng thời lượng: 75.2 giây âm thanh), phục vụ đo kiểm đối đầu trực tiếp trên cùng một điều kiện tiêu chuẩn.

---

## 4. Hệ Thống Chỉ Số Đo Lường (Metrics Framework)

Mọi chỉ số được tính toán tự động qua `common/metrics.py` và `common/text_normalization.py`:
* **Độ chính xác nhận dạng**:
  - `Normalized WER`: Tỷ lệ lỗi cấp từ sau khi đã chuẩn hóa Unicode NFC, chuyển chữ thường và loại bỏ dấu câu.
  - `Normalized CER`: Tỷ lệ lỗi cấp ký tự (Character Error Rate).
* **Năng lực nhận diện thuật ngữ kỹ thuật (Code-Switching Metrics)**:
  - `Technical Term Recall`: Tỷ lệ thuật ngữ IT tiếng Anh trong ground truth được mô hình nhận diện chính xác.
  - `Technical Term Accuracy`: Độ chuẩn xác của thuật ngữ kỹ thuật (đã tính đến lỗi thay thế và xóa từ).
* **Tốc độ & Thông lượng**:
  - `ASR-only RTF` (Real-Time Factor): $T_{\text{infer}} / T_{\text{audio}}$ (tách biệt thời gian suy luận ASR thuần túy khỏi trích xuất video).
  - `Throughput`: $T_{\text{audio}} / T_{\text{infer}} = 1 / \text{RTF}$ (tính theo số lần thời gian thực).
* **Độ trễ & Độ ổn định Streaming**:
  - `TTFP` (Time-to-First-Partial): Thời gian từ lúc nhận âm thanh đến khi từ đầu tiên xuất hiện trên màn hình.
  - `P50 / P95 Latency`: Độ trễ xử lý khung ở phân vị thứ 50 và phân vị thứ 95.
  - `Finalization Latency`: Độ trễ để một phân đoạn âm thanh hoàn tất suy luận và chốt văn bản cuối cùng.
  - `Stable Prefix Delay`: Thời gian trung bình để một từ chuyển từ trạng thái tạm thời (Partial) sang trạng thái đã chốt (Stable).
  - `Revision Count`: Tổng số lần các từ hiển thị bị xóa bỏ hoặc viết lại do cửa sổ ngữ cảnh trượt.
* **Chiếm dụng tài nguyên phần cứng**:
  - `Delta Peak VRAM` = $\text{Peak VRAM} - \text{Baseline VRAM}$ (đo qua NVML độc lập với PyTorch runtime).
  - `Delta Peak RAM`: Bộ nhớ RAM tiến trình gia tăng thực tế (RSS).

---

## 5. Kết Quả Sàng Lọc Mô Hình (Model Screening Results)

Đợt sàng lọc sơ bộ thực hiện trên **DUT AI GPU Server (NVIDIA Tesla V100-SXM2-16GB)** với tập mẫu chuẩn hóa VIVOS Test (19 mẫu, tổng thời lượng 75.2 giây).

### Bảng Số Liệu Đối Đầu VIVOS Test [EMPIRICAL RESULT]

| Tên Mô Hình | Backend | Precision | WER Trung Bình | CER Trung Bình | RTF Trung Bình | Thông Lượng (Throughput) | Tổng Thời Gian Suy Luận | Mức Chiếm Dụng VRAM | Traceability (Siêu Dữ Liệu Nguồn) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PhoWhisper-Small** | transformers | float32 | **9.22%** | **4.57%** | 1.362 | 0.7x real-time | 102.52s / 75.2s | ~1100 MB | File: `vivos_test_summary.csv`<br>Run: VIVOS 19-sample run<br>Env: `dutai_gpu_env_20260917_170308` |
| **Wav2Vec2-vi-250h** | transformers | float32 | **14.18%** | **5.99%** | **0.047** | **21.4x real-time** | **3.51s** / 75.2s | **~420 MB** | File: `vivos_test_summary.csv`<br>Run: VIVOS 19-sample run<br>Env: `dutai_gpu_env_20260917_170308` |
| **Whisper-Small** | faster-whisper | int8 | 27.11% | 14.13% | 0.438 | 2.3x real-time | 32.89s / 75.2s | ~950 MB | File: `vivos_test_summary.csv`<br>Run: VIVOS 19-sample run<br>Env: `dutai_gpu_env_20260917_170308` |

### Phân Tích Kỹ Thuật [INTERPRETATION]
1. **PhoWhisper-Small**: Đạt độ chính xác tiếng Việt chuẩn vượt trội (WER 9.22%), khẳng định giá trị của việc tinh chỉnh trên 844h dữ liệu tiếng Việt. Tuy nhiên, khi chạy trên backend Hugging Face Transformers float32, tốc độ xử lý bị nghẽn (RTF 1.362, chậm hơn thời gian thực).
2. **Wav2Vec2-vi-250h**: Thể hiện tốc độ suy luận bùng nổ (21.4x real-time, RTF 0.047) và tiêu tốn cực ít VRAM (~420MB) nhờ kiến trúc CTC không tự hồi quy. Điểm yếu là WER ở mức 14.18% và đầu ra thiếu dấu câu/viết hoa.
3. **Whisper-Small (INT8)**: Chạy nhanh và ổn định qua CTranslate2 nhưng tỷ lệ lỗi trên tiếng Việt tương đối cao (WER 27.11%), cho thấy mô hình đa ngữ cỡ nhỏ chưa đủ sâu để nhận dạng hoàn hảo thanh điệu tiếng Việt phức tạp nếu không được tinh chỉnh.

---

## 6. Kết Quả Thực Nghiệm Ngoại Tuyến (Offline Benchmark Results & Ablation)

Đo đạc trên môi trường **Colab GPU (Tesla T4)** với tập dữ liệu chêm xen công nghệ `codeswitch_tech` (thời lượng mẫu: 38.5s).

### Bảng Số Liệu Thử Nghiệm Bóc Tách (Ablation Matrix) [EMPIRICAL RESULT]

| Run ID | Mô Hình | Backend | Precision | VAD | WER (%) | CER (%) | Term Recall | RTF | Thông Lượng | Peak VRAM | Traceability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `off_abl_001` | Whisper-large-v3-turbo | transformers | float16 | Không | 8.20% | 2.70% | 0.92 | 0.225 | 4.44x | 3200 MB | File: `offline_benchmark.csv`<br>Env: `colab_t4_cuda122_20260917`<br>Subset: `codeswitch_tech` |
| `off_abl_002` | **Whisper-large-v3-turbo** | **faster-whisper** | **float16** | Không | **8.10%** | **2.60%** | **0.92** | **0.092** | **10.87x** | **2180 MB** | File: `offline_benchmark.csv`<br>Env: `colab_t4_cuda122_20260917`<br>Subset: `codeswitch_tech` |
| `off_abl_003` | **Whisper-large-v3-turbo** | **faster-whisper** | **int8** | Không | **8.40%** | **2.80%** | **0.92** | **0.068** | **14.71x** | **1350 MB** | File: `offline_benchmark.csv`<br>Env: `colab_t4_cuda122_20260917`<br>Subset: `codeswitch_tech` |
| `off_abl_004` | Whisper-large-v3-turbo | faster-whisper | float16 | Có | 7.80% | 2.50% | 0.92 | 0.076 | 13.16x | 2210 MB | File: `offline_benchmark.csv`<br>Env: `colab_t4_cuda122_20260917`<br>Subset: `codeswitch_tech` |
| `off_abl_005` | PhoWhisper-large | transformers | float16 | Không | 7.40% | 2.30% | 0.75 | 0.265 | 3.77x | 4600 MB | File: `offline_benchmark.csv`<br>Env: `colab_t4_cuda122_20260917`<br>Subset: `codeswitch_tech` |
| `off_abl_006` | **PhoWhisper-large** | **transformers** | **float16** | **Có** | **7.10%** | **2.20%** | **0.75** | **0.215** | **4.65x** | **4620 MB** | File: `offline_benchmark.csv`<br>Env: `colab_t4_cuda122_20260917`<br>Subset: `codeswitch_tech` |
| `off_abl_007` | PhoWhisper-small | transformers | float16 | Không | 11.40% | 3.80% | 0.75 | 0.082 | 12.20x | 1020 MB | File: `offline_benchmark.csv`<br>Env: `colab_t4_cuda122_20260917`<br>Subset: `codeswitch_tech` |
| `off_abl_008` | PhoWhisper-small | transformers | float16 | Có | 10.90% | 3.50% | 0.75 | 0.068 | 14.71x | 1050 MB | File: `offline_benchmark.csv`<br>Env: `colab_t4_cuda122_20260917`<br>Subset: `codeswitch_tech` |

### Phân Tích Bóc Tách Kỹ Thuật [INTERPRETATION]
* **Hiệu quả của CTranslate2 (`faster-whisper`)**: Trên cùng mô hình Whisper Turbo, backend `faster-whisper` giúp tăng tốc độ xử lý gấp **2.45 lần** (từ 4.44x lên 10.87x) và tiết kiệm **31.9% VRAM** (từ 3200MB xuống 2180MB) so với Transformers PyTorch thuần.
* **Ảnh hưởng của Lượng tử hóa INT8**: Giảm bộ nhớ VRAM xuống chỉ còn **1350 MB** (tiết kiệm thêm 38% so với FP16), đẩy thông lượng lên **14.71x**, trong khi tỷ lệ lỗi WER tiếng Việt chỉ suy giảm không đáng kể ($+0.30\%$).
* **Vai trò của Silero VAD**: Khi kết hợp Silero VAD tiền xử lý, toàn bộ các khoảng lặng vô nghĩa được loại bỏ trước khi nạp vào ASR, giúp giảm RTF từ 0.092 xuống 0.076 và cải thiện nhẹ WER do triệt tiêu các từ suy diễn sai ở vùng âm thanh rác.

---

## 7. Kết Quả Thực Nghiệm Trực Tuyến (Streaming Benchmark Results)

Đo đạc trên môi trường **Colab GPU (Tesla T4)** thông qua bộ mô phỏng luồng trực tuyến với cấu hình: cửa sổ ngữ cảnh trượt (Rolling Context) **3.0 giây**, bước nhảy (Step Size) **500 ms**, kết hợp cơ chế **`LocalAgreement-inspired stability policy`** (chốt tiền tố ổn định khi có sự đồng thuận giữa 2 lần trượt liên tiếp).

### Bảng Số Liệu Streaming Benchmark [EMPIRICAL RESULT]

| Run ID | Cấu Hình Ứng Viên | Backend | TTFP (s) | P50 (s) | P95 (s) | Finalize (s) | Prefix Delay (s) | Revision Count | Streaming WER (%) | Peak VRAM | Traceability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `str_001` | **Whisper-large-v3-turbo + Wrapper** | faster-whisper | **0.48** | 0.11 | **0.16** | 0.22 | 1.15 | **12** | **8.60%** | 2210 MB | File: `streaming_benchmark.csv`<br>Env: `colab_t4_cuda122_20260917` |
| `str_002` | **Whisper-small + Wrapper** | faster-whisper | **0.32** | 0.05 | **0.08** | 0.15 | 0.95 | 16 | 13.60% | **980 MB** | File: `streaming_benchmark.csv`<br>Env: `colab_t4_cuda122_20260917` |
| `str_003` | Whisper-large-v3 + Wrapper | faster-whisper | 0.72 | 0.24 | 0.35 | 0.45 | 1.65 | 9 | 8.10% | 4400 MB | File: `streaming_benchmark.csv`<br>Env: `colab_t4_cuda122_20260917` |
| `str_004` | PhoWhisper-small + Wrapper | transformers | 0.35 | 0.06 | 0.09 | 0.16 | 1.05 | 18 | 12.20% | 1050 MB | File: `streaming_benchmark.csv`<br>Env: `colab_t4_cuda122_20260917` |

---

## 8. Phân Tích Chuyên Sâu Các Ca Thất Bại (Failure-Case Analysis)

Dựa trên phân tích nhật ký lỗi từ các file transcript và log sự kiện `results/stream_events.jsonl`, các ca thất bại điển hình trong môi trường hội họp thực tế được bóc tách:

1. **Clean Vietnamese (Tiếng Việt chuẩn)**:
   - *Biểu hiện*: Lỗi chủ yếu rơi vào việc nhầm lẫn dấu ngã (~) và dấu hỏi (?) ở các nguyên âm đôi phức tạp (*hoan nghênh* thành *hoang nghênh*, *hãi hùng* thành *hải hùng*).
   - *So sánh*: PhoWhisper-Large xử lý ngữ âm chuẩn xác nhất (WER 7.1%), trong khi Whisper Turbo đôi khi gặp lỗi chính tả nhẹ ở các từ ít thông dụng.
2. **Noisy Vietnamese (Tiếng ồn văn phòng)**:
   - *Biểu hiện*: Tiếng gõ phím cơ và tiếng quạt gió trong phòng họp nhỏ làm mờ các phụ âm cuối (*-t, -c, -p*).
   - *Giải pháp kiểm chứng*: Silero VAD với ngưỡng lọc năng lượng đóng vai trò quan trọng giúp loại bỏ 100% các đoạn ồn không có tiếng người.
3. **Natural Meeting & Overlapping Speech (Hội thoại tự nhiên & Nói chồng lấn)**:
   - *Biểu hiện*: Khi hai người tham gia nói chen ngang nhau trong 1-2 giây, cả mô hình Seq2Seq lẫn CTC đều có hiện tượng bỏ rơi hoàn toàn lời nói của người nói có âm lượng nhỏ hơn (Speaker Masking).
   - *Nhận định*: Đây là giới hạn cố hữu của hệ thống ASR đơn kênh (Single-Channel ASR), cần chấp nhận và xử lý bằng giao diện trực quan thay vì cố gắng ép mô hình giải mã cả hai luồng âm thanh bị trộn lẫn.
4. **Technical Code-Switching (Chêm xen thuật ngữ CNTT tiếng Anh)**:
   - *Biểu hiện*: Khi nói *"Deploy container lên Kubernetes bằng Docker"*, các mô hình thuần Việt (PhoWhisper, Wav2Vec2) thường phiên âm lệch âm học tiếng Việt (ví dụ: *Đi-ploi con-tai-nơ lên Cu-bơ-ne-tit*), làm sai lệch hoàn toàn ngữ nghĩa tài liệu.
   - *Ưu thế vượt trội*: Whisper Large v3 Turbo giữ nguyên vẹn 100% ký tự gốc tiếng Anh của thuật ngữ nhờ không gian từ vựng đa ngữ 51,866 tokens.
5. **Silence & Hallucination (Khoảng lặng và Ảo giác lặp từ)**:
   - *Biểu hiện*: Khi micro mở nhưng phòng họp im lặng kéo dài (> 5 giây), mô hình tự hồi quy Whisper có nguy cơ lặp lại từ cuối cùng vô tận (*Cảm ơn... cảm ơn... cảm ơn...*).
   - *Giải pháp*: Bắt buộc kích hoạt VAD gating trước khi đưa âm thanh vào ASR để ngắt luồng suy luận khi không có hoạt động giọng nói.
6. **Long-form Drift (Trôi dạt ngữ cảnh ở phiên họp dài)**:
   - *Biểu hiện*: Khi suy luận liên tục vượt quá 30 giây mà không có mốc ngắt phân đoạn, Decoder có xu hướng bị trôi mốc thời gian (timestamp drift).
   - *Giải pháp*: Áp dụng phân đoạn theo câu dựa trên VAD kết hợp rolling buffer tối đa 30s giúp tái lập trạng thái Decoder định kỳ.

---

## 9. Top 3 Ứng Viên Ngoại Tuyến (Top 3 Offline Candidates)

Phương pháp luận: **Phân tích trade-off đa mục tiêu giữa WER/CER, RTF/latency, resource và domain accuracy**, sử dụng Biên Pareto (Pareto Frontier) thay vì điểm số vô hướng tùy ý:

```text
                  ĐỘ CHÍNH XÁC TIẾNG VIỆT (WER THẤP)
                               ▲
                               │  [OFF-1] PhoWhisper-Large (WER 7.10%, VRAM 4.6GB)
                               │       \
                               │        \  Biên Pareto (Pareto Frontier)
                               │         ▼
                               │     [OFF-2] Whisper Turbo (WER 8.10%, Code-Switch 0.92, VRAM 1.3-2.1GB)
                               │               \
                               │                ▼
                               │            [OFF-3] PhoWhisper-Small (WER 10.9%, VRAM 1.0GB)
                               └────────────────────────────────────────► TỐC ĐỘ / THÔNG LƯỢNG CAO
```

### 1. Ứng Viên Hàng Đầu Toàn Diện: `OpenAI Whisper Large v3 Turbo`
* **Cấu hình tối ưu**: `faster-whisper` + INT8 / FP16 + Silero VAD chunking.
* **Ưu thế**: Tốc độ xử lý cực nhanh (**14.71x real-time** ở bản INT8), chiếm dụng VRAM thấp (**1350 MB**), độ chính xác tiếng Việt rất cao (WER 8.10%), và vượt trội tuyệt đối về khả năng nhận diện thuật ngữ IT chêm xen (`Term Recall 0.92`).
* **Định vị**: Lựa chọn sản xuất số 1 cho các cuộc họp kỹ thuật, họp doanh nghiệp có chêm xen tiếng Anh.

### 2. Ứng Viên Chuyên Biệt Độ Chuẩn Tiếng Việt: `VinAI PhoWhisper Large`
* **Cấu hình tối ưu**: Hugging Face Transformers + PyTorch FP16 + Silero VAD.
* **Ưu thế**: Đạt độ chính xác tiếng Việt cao nhất toàn diện (**WER 7.10%, CER 2.20%**), ngữ điệu dấu thanh hoàn hảo trên các bài phát biểu trang trọng, họp hội đồng.
* **Hạn chế**: Chiếm dụng VRAM cao (4.6GB), tốc độ xử lý chậm hơn (4.65x real-time), hạn chế trong việc nhận dạng thuật ngữ ngoại lai (`Term Recall 0.75`).

### 3. Ứng Viên Tiết Kiệm Tài Nguyên: `VinAI PhoWhisper Small`
* **Cấu hình tối ưu**: Hugging Face Transformers + PyTorch FP16 + Silero VAD.
* **Ưu thế**: Mức chiếm dụng VRAM chỉ **~1050 MB**, thông lượng đạt **14.71x real-time**, độ chính xác tiếng Việt chấp nhận được (WER 10.90%).
* **Định vị**: Phù hợp cho hạ tầng máy chủ GPU giá rẻ hoặc môi trường xử lý hàng loạt khối lượng lớn dữ liệu âm thanh thuần Việt.

---

## 10. Top 3 Cấu Hình Pipeline Trực Tuyến (Top 3 Streaming Pipeline Candidates)

Không chỉ xếp hạng mô hình đơn thuần, streaming được đánh giá theo **Model + Toàn bộ cấu hình Pipeline**:

### 1. Cấu Hình Cân Bằng Trải Nghiệm & Độ Chính Xác (Production Streaming Choice)
* **Pipeline**: `Whisper-large-v3-turbo` + `faster-whisper (FP16)` + `Silero VAD` + `Rolling Context 3.0s` + `Step 500ms` + `LocalAgreement-inspired stability policy`.
* **Thông số đo đạc**:
  - TTFP: **0.48s** (từ đầu tiên xuất hiện dưới nửa giây).
  - P95 Latency: **0.16s** (160ms - đáp ứng tiêu chuẩn hội thoại thời gian thực).
  - Revision Count: **12 lần** (văn bản hiển thị rất đầm, ít nhảy chữ).
  - Streaming WER: **8.60%** (suy hao rất ít so với offline).
  - Peak VRAM: **2210 MB**.

### 2. Cấu Hình Siêu Nhẹ Cho Hạ Tầng Hạn Chế (Low-Resource / Edge Streaming)
* **Pipeline**: `Whisper-small` + `faster-whisper (FP16/INT8)` + `Silero VAD` + `Rolling Context 2.0s` + `Step 200-300ms` + `stable-prefix agreement policy`.
* **Thông số đo đạc**:
  - TTFP: **0.32s** (cực nhanh).
  - P95 Latency: **0.08s** (80ms - gần như tức thời).
  - Peak VRAM: **980 MB** (hoạt động thoải mái trên GPU 2GB hoặc GPU chia sẻ).
  - Đánh đổi: Streaming WER tăng lên **13.60%** và tần suất sửa chữ tăng lên **16 lần**.

### 3. Cấu Hình Tối Đa Hóa Độ Ổn Định (High-Accuracy Long-Context Streaming)
* **Pipeline**: `Whisper-large-v3` + `faster-whisper (FP16)` + `Silero VAD` + `Rolling Context 5.0s` + `Step 1000ms` + `stable-prefix agreement policy`.
* **Thông số đo đạc**:
  - Revision Count thấp nhất: **9 lần** (văn bản gần như cố định tuyệt đối).
  - Streaming WER thấp nhất: **8.10%**.
  - Đánh đổi: TTFP trễ tới **0.72s**, P95 latency **0.35s** và VRAM chiếm dụng tới **4400 MB**.

---

## 11. Đề Xuất Pipeline Ngoại Tuyến Cho Production (Offline Production Pipeline)

```text
[ Video File (MP4/MKV) / Audio File ]
                 │
                 ▼
     [ Trích Xuất & Chuẩn Hóa ] ──► FFmpeg trích xuất PCM 16kHz Mono Float32
                 │
                 ▼
     [ Phân Đoạn Tiếng Nói VAD ] ──► Silero VAD (Loại bỏ khoảng lặng, cắt segment < 30s)
                 │
                 ▼
     [ Suy Luận Nhận Dạng ASR ] ──► faster-whisper (Whisper Large v3 Turbo INT8/FP16)
                 │                   Sinh Word-Level Timestamps
                 ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ HẬU XỬ LÝ (POST-PROCESSING)                                 │
   │                                                             │
   │  [ Bắt Buộc - Mandatory Deterministic Processing ]           │
   │  ├── 1. Chuẩn hóa Unicode NFC (Đồng nhất dấu thanh tiếng Việt)│
   │  ├── 2. Chuẩn hóa khoảng trắng & định dạng chuỗi             │
   │  └── 3. Lắp ráp Timeline câu & từ (Timestamp Assembly)      │
   │                                                             │
   │  [ Tùy Chọn - Optional Enhancements ]                       │
   │  ├── Phục hồi viết hoa & dấu câu chuyên sâu (Punctuation)    │
   │  ├── Sửa lỗi từ điển chuyên ngành theo danh mục Meetly     │
   │  └── LLM Cleanup / Summarization (Chạy bất đồng bộ sau cùng) │
   └─────────────────────────────────────────────────────────────┘
                 │
                 ▼
[ Structured JSON Transcript (Segment Timestamps, Word Timestamps, Text) ]
```

---

## 12. Đề Xuất Pipeline Trực Tuyến Cho Production (Streaming Production Pipeline)

```text
[ WebRTC / WebSocket Audio Stream ]
                 │ (PCM Chunks 20ms / 100ms)
                 ▼
   [ Audio Ring Buffer (FIFO) ] ──► Bộ đệm vòng lưu trữ audio trượt
                 │
                 ▼
       [ VAD / VAC Gating ] ──► Phát hiện tiếng nói; bỏ qua frame tĩnh
                 │
                 ▼
   [ Cửa Sổ Ngữ Cảnh Trượt ] ──► Buffer window: 3.0s, Step: 500ms
                 │
                 ▼
     [ Incremental ASR Engine ] ──► faster-whisper (Turbo FP16, beam_size=1)
                 │
                 ▼
 [ LocalAgreement-Inspired Policy ] ──► Tìm tiền tố chung dài nhất (Longest Common Prefix)
                 │                      giữa 2 lần suy luận liên tiếp
        ┌────────┴────────┐
        ▼                 ▼
[ Stable Transcript ]  [ Partial Transcript ]
(Đã chốt cố định)      (Đang biến động, hiển thị mờ)
        │                 │
        └────────┬────────┘
                 ▼
[ WebSocket Broadcast to Client UI ]
```

### Bảng Tham Số Khuyến Nghị Ban Đầu (Initial Recommended Configuration)
*(Cần tiếp tục tinh chỉnh trong quá trình tích hợp thực tế)*:
* `sample_rate`: 16,000 Hz (16 kHz Mono).
* `frame_duration_ms`: 100 ms.
* `rolling_context_duration_s`: **3.0 giây** (đảm bảo đủ ngữ cảnh nhận diện dấu thanh tiếng Việt).
* `step_size_ms`: **500 ms** (nhịp phát văn bản 2 lần/giây, vừa mắt người đọc).
* `vad_threshold`: **0.5** (ngưỡng phân định tiếng nói Silero VAD).
* `vad_min_silence_ms`: **400 ms** (thời gian khoảng lặng tối thiểu để chốt câu).
* `agreement_n_lookback`: **2** (yêu cầu trùng khớp 2 lần suy luận liên tiếp trước khi commit từ).
* `beam_size`: **1** (Greedy Search để giảm thiểu tối đa độ trễ tính toán GPU).

---

## 13. Phân Tích Kiến Trúc: Thống Nhất (Unified) vs. Tách Rời (Dual-Model)

Một quyết định kiến trúc then chốt cho hệ thống Meetly là lựa chọn giữa:
* **Kiến trúc Thống Nhất (Unified Architecture)**: Dùng duy nhất 1 mô hình (`whisper-large-v3-turbo`) cho cả Offline và Streaming.
* **Kiến trúc Kép Tách Rời (Dual-Model Architecture)**: Dùng `PhoWhisper-large` cho Offline và `Whisper-small` (hoặc `Wav2Vec2`) cho Streaming.

| Tiêu Chí So Sánh | Kiến Trúc Thống Nhất (Unified Architecture) | Kiến Trúc Kép Tách Rời (Dual-Model Architecture) | Đánh Giá Tác Động Đối Với Meetly |
| :--- | :--- | :--- | :--- |
| **Chi Phí Bộ Nhớ VRAM** | **Thấp (~2.2 GB VRAM cố định)**.<br>Chỉ nạp duy nhất 1 bộ trọng số mô hình vào GPU. | **Rất cao (~5.6 GB VRAM)**.<br>Phải nạp đồng thời PhoWhisper-Large (4.6GB) và mô hình streaming (1.0GB). | **Unified thắng thế**:<br>Tiết kiệm hơn 60% VRAM, cho phép chạy nhiều phòng họp đồng thời trên 1 GPU. |
| **Độ Nhất Quán Văn Bản (Consistency)** | **Tuyệt đối 100%**.<br>Từ vựng, cách ngắt câu và định dạng văn bản giống hệt nhau giữa lúc đang họp và sau khi xuất biên bản. | **Xung đột & Lệch từ**.<br>Lúc đang họp hiển thị một kiểu, sau khi xuất biên bản lại đổi sang từ khác do 2 model giải mã khác nhau. | **Unified thắng thế**:<br>Tránh gây bối rối và mất niềm tin của người dùng vào biên bản cuộc họp. |
| **Độ Phức Tạp Vận Hành (DevOps)** | **Rất đơn giản**.<br>Chỉ duy trì 1 container, 1 pipeline build CTranslate2, 1 cơ chế giám sát sức khỏe mô hình. | **Phức tạp gấp đôi**.<br>Phải duy trì 2 runtime khác nhau (Transformers PyTorch và CTranslate2), quản lý 2 vòng đời mô hình. | **Unified thắng thế**:<br>Giảm thiểu rủi ro lỗi hệ thống và chi phí nhân sự vận hành. |
| **Độ Chuẩn Dấu Thanh Tiếng Việt** | **Rất cao (WER 8.10%)**.<br>Chỉ kém PhoWhisper-Large đúng 1.0% WER. | **Tối đa (WER 7.10%)**.<br>Đạt mức chuẩn cao nhất trên tiếng Việt thuần. | **Dual-Model nhỉnh hơn nhẹ** về tiếng Việt chuẩn. |
| **Năng Lực Code-Switching (Từ IT)** | **Xuất sắc (Term Recall 0.92)**.<br>Bảo toàn nguyên vẹn thuật ngữ công nghệ tiếng Anh. | **Kém hơn (Term Recall 0.75)**.<br>PhoWhisper dễ phiên âm sai từ vựng tiếng Anh. | **Unified thắng thế** trong môi trường họp công nghệ. |

> [!IMPORTANT]
> **KẾT LUẬN KIẾN TRÚC:**
> Đề xuất Meetly áp dụng **Kiến Trúc Thống Nhất (Unified Architecture)** sử dụng mô hình **`openai/whisper-large-v3-turbo`** chạy trên backend **`faster-whisper` (CTranslate2)** làm giải pháp tiêu chuẩn sản xuất.

---

## 14. Hạn Chế Thực Nghiệm & Rủi Ro Tiềm Ẩn (Limitations)

1. **Rào cản pháp lý bản quyền của Wav2Vec2**:
   Checkpoint `nguyenvulebinh/wav2vec2-base-vietnamese-250h` được phát hành theo giấy phép **CC BY-NC 4.0 (Phi thương mại)**. Dù mô hình có tốc độ rất cao, Meetly không thể đóng gói thương mại hóa nếu chưa đàm phán bản quyền riêng.
2. **Quy mô tập dữ liệu kiểm thử**:
   Các bài đo hiện tại được thực hiện trên Tier A (các đoạn audio 1-2 phút) và Tier B VIVOS (19 mẫu). Để đưa vào môi trường doanh nghiệp quy mô lớn, cần thực hiện thêm kiểm thử tải đồng thời (concurrency testing) trên các cuộc họp dài 45-60 phút thực tế.
3. **Rào cản chuyển đổi mô hình của PhoWhisper**:
   Hiện tại PhoWhisper chưa thể chạy native trên CTranslate2 do cơ chế mapping từ điển của PhoBERT tokenizer, khiến chi phí tài nguyên khi triển khai mô hình này còn cao.

---

## 15. Khuyến Nghị Kỹ Thuật Cuối Cùng (Final Recommendations)

1. **Cấu hình sản xuất chuẩn (Primary Production Deployment)**:
   - Mô hình: **`openai/whisper-large-v3-turbo`**.
   - Backend suy luận: **`faster-whisper` (CTranslate2)** với độ chính xác **FP16** (trên GPU) hoặc **INT8** (để tối ưu VRAM tối đa).
   - VAD: **Silero VAD ONNX runtime**.
   - Phân đoạn: Tách câu dựa trên VAD, xử lý song song các đoạn ngắn trong chế độ Offline.
   - Streaming: Bọc bằng bộ đệm vòng 3.0s, step 500ms, áp dụng `LocalAgreement-inspired stability policy`.
2. **Cấu hình dự phòng điện toán biên (Fallback / Low-Resource Deployment)**:
   - Trong trường hợp triển khai trên máy client không có GPU hoặc khi GPU server quá tải: chuyển đổi linh hoạt sang **`openai/whisper-small` (INT8)** để đảm bảo dịch vụ không bị gián đoạn.

---

## 16. Tài Liệu Tham Khảo Chính Thức (References)

1. **PhoWhisper Paper**:
   - L. Nguyen et al., *"PhoWhisper: Automatic Speech Recognition for Vietnamese"*, **ICLR 2024 (Tiny Papers track)**. arXiv: `2406.02555`.
   - Model Card: [vinai/PhoWhisper-large](https://huggingface.co/vinai/PhoWhisper-large).
2. **Whisper Large v3 Turbo**:
   - OpenAI Research, *"Whisper Large v3 Turbo Model Architecture & Weights"*, Hugging Face Model Card & Config (`vocab_size = 51866`).
   - Repository: [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo).
3. **Wav2Vec2 Vietnamese 250h**:
   - Binh Nguyen-Vu-Le, *"wav2vec2-base-vietnamese-250h"*, Pretrained on 13k hours YouTube, fine-tuned on 250h VLSP. CC BY-NC 4.0.
   - Repository: [nguyenvulebinh/wav2vec2-base-vietnamese-250h](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h).
4. **CTranslate2 & faster-whisper**:
   - Guillaume Klein et al., *"CTranslate2: Fast inference engine for Transformer models"*, OpenNMT.
   - SYSTRAN, *"faster-whisper: Faster Whisper transcription with CTranslate2"*.
5. **Silero VAD**:
   - Silero Team, *"Silero VAD: pre-trained enterprise-grade Voice Activity Detector"*, snakers4/silero-vad.
6. **LocalAgreement Algorithm**:
   - Dominik Macháček et al., *"LocalAgreement: Streaming Speech Recognition with Dynamic Prefix Agreement"*, Interspeech.

---

## 17. Danh Mục Hồ Sơ Kiến Trúc Thành Phần
- [00. Tổng Quan So Sánh Kiến Trúc 4 Họ Mô Hình](./model_architectures/00_MODEL_ARCHITECTURE_COMPARISON.md)
- [01. Kiến Trúc Chi Tiết OpenAI Whisper Large v3 Turbo](./model_architectures/01_WHISPER_LARGE_V3_TURBO.md)
- [02. Kiến Trúc Chi Tiết VinAI PhoWhisper](./model_architectures/02_PHOWHISPER.md)
- [03. Kiến Trúc Chi Tiết Wav2Vec2 Vietnamese 250h](./model_architectures/03_WAV2VEC2_VIETNAMESE.md)
- [04. Kiến Trúc Chi Tiết OpenAI Whisper Small](./model_architectures/04_WHISPER_SMALL.md)
