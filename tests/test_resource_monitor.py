"""
Kiểm thử tự động cho module giám sát tài nguyên (common/resource_monitor.py).
"""

import time
import pytest
from common.resource_monitor import ResourceMonitor


class TestResourceMonitor:

    def test_resource_monitor_context_lifecycle(self):
        """Kiểm tra vòng đời của ResourceMonitor trong context manager."""
        with ResourceMonitor(sample_interval_s=0.01) as monitor:
            # Thực hiện tác vụ nhỏ giả lập
            data = [i for i in range(100_000)]
            time.sleep(0.05)
            del data

        metrics = monitor.get_metrics()

        assert "gpu_memory_source" in metrics
        assert metrics["gpu_memory_source"] in ("nvml", "torch", "unavailable")

        # Kiểm tra tính hợp lệ của RAM
        assert metrics["baseline_ram_mb"] > 0.0
        assert metrics["peak_ram_mb"] >= metrics["baseline_ram_mb"]
        assert metrics["delta_peak_ram_mb"] >= 0.0

        # Kiểm tra tính hợp lệ của VRAM
        assert metrics["baseline_vram_mb"] >= 0.0
        assert metrics["peak_vram_mb"] >= metrics["baseline_vram_mb"]
        assert metrics["delta_peak_vram_mb"] >= 0.0

        # Kiểm tra CPU
        assert metrics["avg_cpu_percent"] >= 0.0
