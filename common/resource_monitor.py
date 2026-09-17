"""
Module giám sát tài nguyên phần cứng độc lập backend (Backend-Agnostic Resource Monitoring).

Cung cấp:
- Giám sát bộ nhớ 3 thành phần: Baseline, Peak và Delta (Peak - Baseline) cho cả VRAM và RAM.
- Giám sát VRAM độc lập qua NVML (hỗ trợ chính xác cho cả CTranslate2 và PyTorch allocator).
- Giám sát RAM tiến trình (RSS) và mức tải CPU định kỳ qua thread nền.
- Tuyệt đối KHÔNG gọi torch.cuda.empty_cache() trong quá trình lấy mẫu để tránh làm méo độ trễ.
- Tự động fallback mượt mà trên môi trường CPU (không có CUDA).
"""

import time
import threading
import psutil
from typing import TypedDict

# Kiểm tra tính khả dụng của PyTorch CUDA
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Kiểm tra tính khả dụng của pynvml (NVIDIA Management Library)
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    _nvml_device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    NVML_AVAILABLE = False
    _nvml_device_handle = None


class ResourceSnapshot(TypedDict):
    gpu_memory_source: str
    baseline_vram_mb: float
    peak_vram_mb: float
    delta_peak_vram_mb: float
    baseline_ram_mb: float
    peak_ram_mb: float
    delta_peak_ram_mb: float
    avg_cpu_percent: float
    peak_cpu_percent: float


class ResourceMonitor:
    """
    Context manager giám sát tài nguyên phần cứng trong quá trình thực thi một tác vụ.

    Cách sử dụng:
        with ResourceMonitor() as monitor:
            run_heavy_inference()
        stats = monitor.get_metrics()
    """

    def __init__(self, sample_interval_s: float = 0.05, gpu_index: int | None = None):
        """
        Khởi tạo ResourceMonitor.

        :param sample_interval_s: Khoảng thời gian lấy mẫu định kỳ của thread nền (giây).
        :param gpu_index: Chỉ số GPU cần đo. Mặc định tự động lấy torch.cuda.current_device().
        """
        self.sample_interval_s = sample_interval_s
        self.process = psutil.Process()

        # Xác định chỉ số GPU động
        if gpu_index is not None:
            self.gpu_index = gpu_index
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                self.gpu_index = torch.cuda.current_device()
            except Exception:
                self.gpu_index = 0
        else:
            self.gpu_index = 0

        # Xác định nguồn đo VRAM và nạp handle NVML tương ứng với gpu_index
        self._nvml_handle = None
        if NVML_AVAILABLE:
            try:
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
                self.gpu_memory_source = "nvml"
            except Exception:
                self._nvml_handle = None

        if self._nvml_handle is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.gpu_memory_source = "torch"
            else:
                self.gpu_memory_source = "unavailable"

        # Biến lưu trữ số đo RAM
        self.baseline_ram_mb = 0.0
        self.peak_ram_mb = 0.0

        # Biến lưu trữ số đo VRAM
        self.baseline_vram_mb = 0.0
        self.peak_vram_mb = 0.0

        # Biến lưu trữ số đo CPU
        self.cpu_samples: list[float] = []

        self._running = False
        self._thread: threading.Thread | None = None

    def _get_current_ram_mb(self) -> float:
        """Lấy dung lượng RAM tiến trình thực tế (RSS) tính bằng MB."""
        try:
            rss_bytes = self.process.memory_info().rss
            return round(rss_bytes / (1024.0 * 1024.0), 2)
        except Exception:
            return 0.0

    def _get_current_vram_mb(self) -> float:
        """Lấy dung lượng VRAM đang sử dụng tính bằng MB theo nguồn đo đã chọn."""
        if self.gpu_memory_source == "nvml" and self._nvml_handle is not None:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                return round(mem_info.used / (1024.0 * 1024.0), 2)
            except Exception:
                pass

        if self.gpu_memory_source in ("nvml", "torch") and TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                return round(torch.cuda.memory_allocated(device=self.gpu_index) / (1024.0 * 1024.0), 2)
            except Exception:
                pass

        return 0.0

    def _sample_loop(self) -> None:
        """Vòng lặp lấy mẫu định kỳ chạy trong thread nền."""
        while self._running:
            # Đo RAM
            current_ram = self._get_current_ram_mb()
            if current_ram > self.peak_ram_mb:
                self.peak_ram_mb = current_ram

            # Đo VRAM
            current_vram = self._get_current_vram_mb()
            if current_vram > self.peak_vram_mb:
                self.peak_vram_mb = current_vram

            # Đo CPU
            try:
                cpu = self.process.cpu_percent(interval=None)
                if cpu > 0.0:
                    self.cpu_samples.append(cpu)
            except Exception:
                pass

            time.sleep(self.sample_interval_s)

    def __enter__(self) -> "ResourceMonitor":
        """Bắt đầu đo lường: ghi nhận baseline và khởi động thread giám sát."""
        # Ghi nhận baseline trước khi thực thi
        self.baseline_ram_mb = self._get_current_ram_mb()
        self.peak_ram_mb = self.baseline_ram_mb

        self.baseline_vram_mb = self._get_current_vram_mb()
        self.peak_vram_mb = self.baseline_vram_mb

        self.cpu_samples = []
        # Kích hoạt tính toán cpu_percent ban đầu
        try:
            self.process.cpu_percent(interval=None)
        except Exception:
            pass

        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Dừng đo lường và tính toán tổng kết."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        # Lấy mẫu chốt lần cuối
        final_ram = self._get_current_ram_mb()
        self.peak_ram_mb = max(self.peak_ram_mb, final_ram)

        final_vram = self._get_current_vram_mb()
        self.peak_vram_mb = max(self.peak_vram_mb, final_vram)

    def get_metrics(self) -> ResourceSnapshot:
        """
        Trả về kết quả đo lường tài nguyên phần cứng có cấu trúc.
        """
        delta_vram = max(0.0, round(self.peak_vram_mb - self.baseline_vram_mb, 2))
        delta_ram = max(0.0, round(self.peak_ram_mb - self.baseline_ram_mb, 2))

        avg_cpu = round(sum(self.cpu_samples) / len(self.cpu_samples), 1) if self.cpu_samples else 0.0
        peak_cpu = round(max(self.cpu_samples), 1) if self.cpu_samples else 0.0

        return {
            "gpu_memory_source": self.gpu_memory_source,
            "baseline_vram_mb": self.baseline_vram_mb,
            "peak_vram_mb": self.peak_vram_mb,
            "delta_peak_vram_mb": delta_vram,
            "baseline_ram_mb": self.baseline_ram_mb,
            "peak_ram_mb": self.peak_ram_mb,
            "delta_peak_ram_mb": delta_ram,
            "avg_cpu_percent": avg_cpu,
            "peak_cpu_percent": peak_cpu
        }
