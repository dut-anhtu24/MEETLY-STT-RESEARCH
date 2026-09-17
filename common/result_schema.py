"""
Module định nghĩa cấu trúc dữ liệu kết quả thực nghiệm (Result Schemas).

Sử dụng Pydantic v2 để xác thực kiểu dữ liệu, hỗ trợ xuất JSON Lines (source of truth)
và bảng tổng hợp CSV.
Bao gồm:
- Enum trạng thái thực thi: completed, skipped_oom, skipped_unsupported, failed_runtime.
- Các schema đo lường: Accuracy, Performance, Streaming, Resource, Config Snapshot.
- StreamEventRecord ghi log chi tiết từng event phục vụ phân tích độ ổn định.
- EnvironmentFingerprint ghi nhận thông số môi trường phần cứng và thư viện.
"""

import sys
import platform
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
import psutil
from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    """Trạng thái hoàn thành của một lần chạy benchmark."""
    COMPLETED = "completed"
    SKIPPED_OOM = "skipped_oom"
    SKIPPED_UNSUPPORTED = "skipped_unsupported"
    FAILED_RUNTIME = "failed_runtime"


class AccuracyMetrics(BaseModel):
    """Chỉ số đánh giá độ chính xác nhận dạng giọng nói."""
    normalized_wer: float = Field(..., description="Word Error Rate sau khi chuẩn hóa")
    normalized_cer: float = Field(..., description="Character Error Rate sau khi chuẩn hóa")
    technical_term_recall: float | None = Field(None, description="Tỷ lệ bắt trúng thuật ngữ kỹ thuật")
    technical_term_accuracy: float | None = Field(None, description="Độ chính xác nhận dạng thuật ngữ kỹ thuật")
    gt_word_count: int = Field(0, description="Tổng số từ trong Ground Truth")
    hyp_word_count: int = Field(0, description="Tổng số từ trong Hypothesis")


class PerformanceMetrics(BaseModel):
    """Chỉ số thời gian suy luận và thông lượng xử lý."""
    load_time_s: float | None = Field(None, description="Thời gian nạp model vào RAM/VRAM")
    warmup_time_s: float | None = Field(None, description="Thời gian chạy lượt warm-up đầu tiên")
    asr_inference_time_s: float = Field(..., description="Thời gian suy luận ASR thuần túy (giây)")
    asr_rtf: float = Field(..., description="Real-Time Factor (ASR Time / Audio Duration)")
    throughput_x_realtime: float = Field(..., description="Thông lượng xử lý gấp bao nhiêu lần thời gian thực")
    video_audio_extraction_time_s: float | None = Field(None, description="Thời gian tách audio từ video qua FFmpeg")
    vad_segmentation_time_s: float | None = Field(None, description="Thời gian Silero VAD phân đoạn âm thanh")
    total_end_to_end_time_s: float | None = Field(None, description="Tổng thời gian toàn trình từ đầu vào đến kết quả")


class StreamingMetrics(BaseModel):
    """Chỉ số chuyên biệt cho chế độ trực tuyến / streaming."""
    simulation_mode: str = Field("virtual_clock", description="Chế độ mô phỏng: virtual_clock hoặc wall_clock")
    first_partial_latency_s: float = Field(..., description="Thời gian trễ từ khi có tiếng nói đến partial text đầu tiên")
    p50_processing_latency_s: float = Field(..., description="Độ trễ trung vị P50 của từng bước xử lý frame")
    p95_processing_latency_s: float = Field(..., description="Độ trễ phân vị P95 của từng bước xử lý frame")
    finalization_latency_s: float = Field(..., description="Thời gian từ khi ngừng nói đến khi chốt câu hoàn toàn")
    stable_prefix_delay_s: float = Field(..., description="Độ trễ trung bình để một từ trở thành bất biến")
    revision_count: int = Field(0, description="Tổng số từ/ký tự bị thay đổi hoặc rút lại trước khi chốt")


class ResourceMetrics(BaseModel):
    """Chỉ số tiêu thụ tài nguyên phần cứng (độc lập backend)."""
    gpu_memory_source: str = Field(..., description="Nguồn đo VRAM: nvml, torch, hoặc unavailable")
    baseline_vram_mb: float = Field(0.0, description="VRAM trước khi chạy tác vụ")
    peak_vram_mb: float = Field(0.0, description="VRAM đỉnh điểm trong quá trình chạy")
    delta_peak_vram_mb: float = Field(0.0, description="Dung lượng VRAM thực tế mà model chiếm dụng")
    baseline_ram_mb: float = Field(0.0, description="RAM tiến trình trước khi chạy")
    peak_ram_mb: float = Field(0.0, description="RAM tiến trình đỉnh điểm trong quá trình chạy")
    delta_peak_ram_mb: float = Field(0.0, description="Dung lượng RAM thực tế tăng thêm")
    avg_cpu_percent: float = Field(0.0, description="Mức tải CPU trung bình (%)")
    peak_cpu_percent: float = Field(0.0, description="Mức tải CPU đỉnh điểm (%)")


class ConfigSnapshot(BaseModel):
    """Ảnh chụp cấu hình tham số thực nghiệm."""
    backend: str
    precision: str
    vad_enabled: bool = False
    vad_threshold: float = 0.5
    word_timestamps: bool = False
    beam_size: int = 5
    warmup_runs: int = 1
    benchmark_runs: int = 3


class BenchmarkRecord(BaseModel):
    """Bản ghi tổng thể của một lần chạy benchmark (Lưu trữ dạng JSON Lines)."""
    run_id: str
    environment_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    mode: str = Field(..., description="Chế độ: offline, streaming, hoặc screening")
    execution_status: ExecutionStatus = Field(ExecutionStatus.COMPLETED)
    model_id: str
    subset: str
    audio_id: str
    audio_duration_s: float
    accuracy: AccuracyMetrics | None = None
    performance: PerformanceMetrics | None = None
    streaming: StreamingMetrics | None = None
    resource: ResourceMetrics | None = None
    config: ConfigSnapshot | None = None
    error_message: str | None = None


class StreamEventRecord(BaseModel):
    """Log sự kiện chi tiết của từng bước suy luận trong streaming."""
    run_id: str
    event_id: int
    timestamp_wall_clock_s: float
    total_audio_received_s: float
    buffer_duration_s: float
    processing_time_s: float
    committed_words_count: int
    partial_words_count: int
    stable_text: str
    partial_text: str
    is_final: bool = False


class EnvironmentFingerprint(BaseModel):
    """Thông tin vân tay môi trường thực thi phần cứng và phần mềm."""
    environment_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    os: str
    python_version: str
    pytorch_version: str
    transformers_version: str | None = None
    ctranslate2_version: str | None = None
    faster_whisper_version: str | None = None
    cuda_available: bool
    cuda_version: str | None = None
    gpu_id: int | str | None = None
    gpu_name: str | None = None
    gpu_total_vram_mb: float = 0.0
    cpu_model: str
    cpu_cores: int
    system_ram_mb: float
    ffmpeg_version: str | None = None
    hf_home: str | None = None

    @classmethod
    def capture(cls, environment_id: str | None = None, gpu_id: int | None = None) -> "EnvironmentFingerprint":
        """Tự động thu thập thông số hệ thống và trả về đối tượng EnvironmentFingerprint."""
        import os
        import subprocess

        # Phiên bản PyTorch & CUDA
        import torch
        pt_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        cuda_ver = torch.version.cuda if cuda_avail else None

        active_gpu_id = gpu_id
        gpu_name = None
        gpu_vram = 0.0

        if cuda_avail:
            try:
                if active_gpu_id is None:
                    active_gpu_id = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(active_gpu_id)
                gpu_vram = round(torch.cuda.get_device_properties(active_gpu_id).total_memory / (1024.0 * 1024.0), 1)
            except Exception:
                pass

        trans_ver = None
        try:
            import transformers
            trans_ver = transformers.__version__
        except ImportError:
            pass

        ct2_ver = None
        try:
            import ctranslate2
            ct2_ver = ctranslate2.__version__
        except ImportError:
            pass

        fw_ver = None
        try:
            import faster_whisper
            fw_ver = faster_whisper.__version__
        except ImportError:
            pass

        # Kiểm tra phiên bản FFmpeg
        ffmpeg_ver = None
        try:
            res = subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            if res.returncode == 0:
                first_line = res.stdout.strip().split("\n")[0]
                ffmpeg_ver = first_line.split()[2] if len(first_line.split()) >= 3 else first_line[:30]
        except Exception:
            ffmpeg_ver = "not_available"

        hf_cache = os.environ.get("HF_HOME", None)

        env_id = environment_id or f"env_{platform.system().lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return cls(
            environment_id=env_id,
            os=f"{platform.system()} {platform.release()} ({platform.machine()})",
            python_version=sys.version.split()[0],
            pytorch_version=pt_ver,
            transformers_version=trans_ver,
            ctranslate2_version=ct2_ver,
            faster_whisper_version=fw_ver,
            cuda_available=cuda_avail,
            cuda_version=cuda_ver,
            gpu_id=active_gpu_id,
            gpu_name=gpu_name,
            gpu_total_vram_mb=gpu_vram,
            cpu_model=platform.processor() or "Unknown CPU",
            cpu_cores=psutil.cpu_count(logical=True) or 1,
            system_ram_mb=round(psutil.virtual_memory().total / (1024.0 * 1024.0), 1),
            ffmpeg_version=ffmpeg_ver,
            hf_home=hf_cache
        )

    def save_to_json(self, file_path: str | Path) -> None:
        """Lưu fingerprint môi trường vào file JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
