"""
Kiểm thử tự động cho Result Schemas và Model Registry (common/result_schema.py, common/model_registry.py).
"""

from pathlib import Path
import pytest
from common.result_schema import (
    ExecutionStatus,
    AccuracyMetrics,
    PerformanceMetrics,
    ResourceMetrics,
    ConfigSnapshot,
    BenchmarkRecord,
    StreamEventRecord,
    EnvironmentFingerprint
)
from common.model_registry import ModelRegistry


class TestSchema:

    def test_benchmark_record_serialization(self):
        """Kiểm tra tạo và chuyển đổi JSON của một BenchmarkRecord chuẩn."""
        record = BenchmarkRecord(
            run_id="test_run_001",
            environment_id="env_test_01",
            mode="offline",
            execution_status=ExecutionStatus.COMPLETED,
            model_id="openai/whisper-small",
            subset="clean_vi",
            audio_id="vi_clean_01",
            audio_duration_s=42.5,
            accuracy=AccuracyMetrics(
                normalized_wer=0.085,
                normalized_cer=0.032,
                technical_term_recall=1.0,
                technical_term_accuracy=1.0,
                gt_word_count=50,
                hyp_word_count=50
            ),
            performance=PerformanceMetrics(
                load_time_s=1.5,
                warmup_time_s=0.5,
                asr_inference_time_s=3.4,
                asr_rtf=0.08,
                throughput_x_realtime=12.5,
                total_end_to_end_time_s=3.8
            ),
            resource=ResourceMetrics(
                gpu_memory_source="unavailable",
                baseline_vram_mb=0.0,
                peak_vram_mb=0.0,
                delta_peak_vram_mb=0.0,
                baseline_ram_mb=500.0,
                peak_ram_mb=750.0,
                delta_peak_ram_mb=250.0,
                avg_cpu_percent=35.0,
                peak_cpu_percent=45.0
            ),
            config=ConfigSnapshot(
                backend="faster-whisper",
                precision="float16",
                vad_enabled=True
            )
        )

        # Kiểm tra serialize ra JSON chuỗi
        json_str = record.model_dump_json()
        assert "test_run_001" in json_str
        assert "completed" in json_str

        # Kiểm tra deserialize ngược lại
        restored = BenchmarkRecord.model_validate_json(json_str)
        assert restored.run_id == record.run_id
        assert restored.execution_status == ExecutionStatus.COMPLETED
        assert restored.accuracy.normalized_wer == 0.085
        assert restored.resource.delta_peak_ram_mb == 250.0

    def test_oom_status_handling(self):
        """Kiểm tra xử lý trạng thái skipped_oom khi model bị tràn bộ nhớ."""
        oom_record = BenchmarkRecord(
            run_id="test_oom_001",
            environment_id="env_test_01",
            mode="offline",
            execution_status=ExecutionStatus.SKIPPED_OOM,
            model_id="vinai/PhoWhisper-large",
            subset="clean_vi",
            audio_id="vi_clean_01",
            audio_duration_s=42.5,
            error_message="CUDA out of memory during inference"
        )
        assert oom_record.execution_status == ExecutionStatus.SKIPPED_OOM
        assert oom_record.accuracy is None
        assert "CUDA out of memory" in oom_record.error_message

    def test_stream_event_record(self):
        """Kiểm tra cấu trúc StreamEventRecord với các trường mới bổ sung."""
        event = StreamEventRecord(
            run_id="test_str_001",
            event_id=5,
            timestamp_wall_clock_s=2.5,
            total_audio_received_s=2.4,
            buffer_duration_s=3.0,
            processing_time_s=0.12,
            committed_words_count=10,
            partial_words_count=4,
            stable_text="chào mọi người",
            partial_text="hôm nay chúng ta",
            is_final=False
        )
        assert event.committed_words_count == 10
        assert event.buffer_duration_s == 3.0
        assert event.is_final is False

    def test_environment_fingerprint_capture(self):
        """Kiểm tra tự động ghi nhận thông tin môi trường."""
        env = EnvironmentFingerprint.capture("env_unit_test")
        assert env.environment_id == "env_unit_test"
        assert env.python_version != ""
        assert env.pytorch_version != ""
        assert env.system_ram_mb > 0.0

    def test_model_registry_and_smoke_isolation(self):
        """Kiểm tra nạp cấu hình models.yaml và lọc tách biệt model smoke-test."""
        models_yaml_path = Path(__file__).parents[1] / "configs" / "models.yaml"
        registry = ModelRegistry(models_yaml_path)

        assert "whisper-small" in registry.catalog
        assert "phowhisper-large" in registry.catalog

        # Kiểm tra metadata
        pho_large = registry.get("phowhisper-large")
        assert pho_large.conversion_required is True
        assert pho_large.compatibility_status == "pending_verification"

        # Kiểm tra loại trừ model smoke-test
        all_candidates = registry.list_candidates(include_smoke_test=False)
        smoke_ids = [m.model_id for m in all_candidates if m.is_smoke_test]
        assert len(smoke_ids) == 0

        # Kiểm tra whisper-tiny được đánh dấu là smoke test
        assert registry.is_smoke_test_model("whisper-tiny-smoke") is True
