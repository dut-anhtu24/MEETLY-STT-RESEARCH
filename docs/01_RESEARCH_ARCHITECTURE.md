# Tài Liệu Kỹ Thuật: Kiến Trúc Nghiên Cứu & Benchmark Framework (Research Architecture)
### Dự án Meetly - Speech-to-Text Subsystem

---

## 1. Định Nghĩa Bài Toán & Hai Chế Độ Độc Lập

Meetly là nền tảng hội họp trực tuyến và phát trực tiếp (livestream) tích hợp chức năng chuyển đổi giọng nói thành văn bản (AI Transcription). 

Trong hệ thống Meetly, bài toán STT không phải là một luồng xử lý đơn lẻ, mà phục vụ **hai chế độ vận hành độc lập** với các ràng buộc kỹ thuật hoàn toàn khác nhau:

```text
                                [ HỆ THỐNG MEETLY ]
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
        [ CHẾ ĐỘ NGOẠI TUYẾN (OFFLINE) ]          [ CHẾ ĐỘ TRỰC TUYẾN (STREAMING) ]
        - Đầu vào: Video / Cuộc họp ghi hình      - Đầu vào: Livestream / Micro trực tiếp
        - Đầu ra: Structured Transcript (JSON)    - Đầu ra: Partial & Stable Transcripts
        - Ưu tiên #1: Độ chính xác (WER/CER)      - Ưu tiên #1: Độ trễ siêu thấp (P95 < 200ms)
        - Ưu tiên #2: Thông lượng cao (RTF << 1)  - Ưu tiên #2: Độ ổn định văn bản (Ít nhấp nháy)
        - Ưu tiên #3: Ngữ cảnh dài không trôi     - Ưu tiên #3: Tiết kiệm tài nguyên cho tải lớn
```

### Vì sao hai chế độ phải được nghiên cứu và thiết kế độc lập?
1. **Khác biệt về cửa sổ ngữ cảnh**: Offline có thể nhìn toàn bộ câu hoặc đoạn 30s để giải mã tự hồi quy chính xác nhất, trong khi Streaming chỉ nhìn thấy các frame âm thanh vừa đến trong cửa sổ trượt ngắn 2-3 giây.
2. **Khác biệt về độ trễ chấp nhận được**: Với video 60 phút, người dùng sẵn sàng chờ 3–5 phút để nhận transcript hoàn chỉnh có word-level timestamps; nhưng với livestream, nếu transcript hiển thị sau lời nói quá 1 giây thì tính năng phụ đề trực tiếp mất hoàn toàn giá trị tương tác.
3. **Tính độc lập của mô hình**: Không ép buộc cả hai chế độ phải dùng chung một model. Quyết định dùng một mô hình thống nhất (Unified) hay hai mô hình riêng biệt (Separate) phải dựa trên **bằng chứng định lượng thực nghiệm**.

---

## 2. Luồng Xử Lý Toàn Trình (Dataflow Pipelines)

### A. Luồng Xử Lý Ngoại Tuyến (Offline Pipeline)
Bắt đầu từ file Video thực tế đến file JSON có cấu trúc hoàn chỉnh:

```text
[ File Video (.mp4/.mkv) ]
       │
       ▼ (FFmpeg Extraction - pcm_s16le, 16kHz, mono)
[ Audio Chuẩn Hóa 16kHz Mono PCM ]
       │
       ▼ (Silero VAD Phân Đoạn Tiếng Nói)
[ Danh Sách Speech Segments ] ──> (Loại bỏ các đoạn im lặng/nhiễu nền)
       │
       ▼ (ASR Engine: faster-whisper FP16/INT8)
[ Danh Sách Từ Kèm Timestamps (Word/Segment Level) ]
       │
       ▼ (Transcript Assembly & Metadata Formatting)
[ Structured JSON File ]
```

### B. Luồng Xử Lý Trực Tuyến (Streaming Pipeline)
Xử lý liên tục luồng frame âm thanh thời gian thực:

```text
[ Live Audio Frames (100ms / 20ms) ]
       │
       ▼
[ FIFO Audio Ring Buffer (Step: 500ms, Context: 3.0s) ]
       │
       ▼
[ Online VAD / VAC Gating ] ──> (Bỏ qua frame tĩnh, phát hiện ngắt câu)
       │
       ▼
[ Incremental ASR Context Decoder ]
       │
       ▼
[ Chính Sách Ổn Định LocalAgreement ]
  ├── Khớp tiền tố chung (Longest Common Prefix) ──> [ STABLE TRANSCRIPT ] (Gửi UI chốt cố định)
  └── Phần đuôi mới biến động                   ──> [ PARTIAL TRANSCRIPT ] (Gửi UI tạm thời)
```

---

## 3. Kiến Trúc Benchmark Framework Dùng Chung

Để đảm bảo tính nhất quán và khả năng tái lập 100%, framework được thiết kế theo cấu trúc module phân lớp:

```text
meetly-stt-research/
├── configs/            # Khai báo cấu hình YAML độc lập
├── data/               # Tập dữ liệu đo kiểm chuẩn hóa Tier A & B
├── common/             # Các module logic nghiệp vụ tái sử dụng:
│   ├── audio.py                 # Chuẩn hóa âm thanh, sinh frame, trích xuất video
│   ├── text_normalization.py   # Chuẩn hóa tiếng Việt 2 lớp & phân tích thuật ngữ IT
│   ├── metrics.py               # WER, CER, RTF, Throughput, Revision Distance
│   ├── resource_monitor.py      # Đo Baseline/Peak/Delta RAM & VRAM (NVML/PyTorch)
│   ├── model_registry.py        # Skeleton quản lý model & cách ly smoke-test
│   ├── result_schema.py         # Pydantic schemas & EnvironmentFingerprint
│   ├── vad.py                   # Silero VAD wrapper với fallback an toàn
│   ├── offline_engine.py        # Động cơ xử lý offline từ video đến JSON
│   └── streaming_engine.py      # Động cơ mô phỏng streaming & LocalAgreement
├── notebooks/          # Notebook orchestration, visualization & nhận xét (Tiếng Việt)
├── results/            # Nguồn dữ liệu gốc (JSON Lines, CSV) và biểu đồ Pareto
└── docs/               # 5 tài liệu nghiên cứu kỹ thuật hoàn chỉnh
```

### Nguyên tắc thiết kế:
- **Notebooks chỉ giữ vai trò Orchestration**: Không copy-paste hàng trăm dòng xử lý vào notebook; toàn bộ thuật toán cốt lõi nằm trong `common/`.
- **JSON Lines là Source of Truth**: Mọi lần chạy benchmark đều lưu chi tiết đầy đủ trong `.jsonl`; file CSV chỉ đóng vai trò bảng tổng hợp đọc nhanh.
- **Không dùng điểm số scalar tùy ý**: Sử dụng **Biên Pareto (Pareto Frontier)** để đánh giá trade-off khách quan giữa Độ chính xác, Tốc độ và Tài nguyên phần cứng.
