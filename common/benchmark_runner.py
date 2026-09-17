"""
Module điều phối thực nghiệm benchmark (Benchmark Runner).

Cung cấp:
- Lớp BenchmarkRunner thực thi đo kiểm mô hình qua các tập dữ liệu.
- Quản lý lượt chạy khởi động (warmup) và các lượt đo chính thức (repeated runs).
- Tích hợp ResourceMonitor đo Baseline, Peak, Delta bộ nhớ RAM và VRAM.
- Tự động bắt lỗi ngoại lệ và phân loại trạng thái: completed, skipped_oom, skipped_unsupported, failed_runtime.
- Tính toán WER, CER, RTF, Throughput và Technical Term Recall/Accuracy.
- Xuất bản ghi ra JSON Lines và bảng tổng hợp CSV.
"""

import time
import csv
import json
from pathlib import Path
from typing import Any, Callable
import pandas as pd

from common.audio import load_audio, get_audio_duration
from common.text_normalization import normalize_vietnamese_text
from common.metrics import (
    compute_wer,
    compute_cer,
    calculate_rtf,
    calculate_throughput,
    calculate_technical_term_metrics
)
from common.resource_monitor import ResourceMonitor
from common.result_schema import (
    BenchmarkRecord,
    ExecutionStatus,
    AccuracyMetrics,
    PerformanceMetrics,
    ResourceMetrics,
    ConfigSnapshot
)


class BenchmarkRunner:
    """Bộ điều phối thực thi các thí nghiệm đo kiểm STT chuẩn hóa."""

    def __init__(
        self,
        environment_id: str,
        warmup_runs: int = 1,
        benchmark_runs: int = 3,
        technical_terms: list[str] | None = None
    ):
        self.environment_id = environment_id
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.technical_terms = technical_terms or []

    def run_single_experiment(
        self,
        run_id: str,
        mode: str,
        model_id: str,
        subset: str,
        audio_id: str,
        audio_path: str | Path,
        ground_truth: str,
        transcribe_fn: Callable[[Any], str],
        load_time_s: float = 0.0,
        config_snapshot: ConfigSnapshot | None = None
    ) -> BenchmarkRecord:
        """
        Thực hiện một lượt đo kiểm độc lập trên một file âm thanh.

        Bao gồm:
        - 1 lượt warm-up (không tính vào thời gian suy luận benchmark).
        - N lượt lặp lại chính thức đo thời gian suy luận (lấy trung bình).
        - Giám sát Baseline/Peak/Delta RAM và VRAM.
        - Xử lý OOM an toàn và xuất bản ghi BenchmarkRecord.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file audio: {audio_path}")

        audio_dur = get_audio_duration(path)
        audio_data, sr = load_audio(path, target_sr=16000)

        # 1. Lượt chạy Warm-up
        warmup_time_s = 0.0
        try:
            t_warmup_0 = time.perf_counter()
            _ = transcribe_fn(audio_data)
            warmup_time_s = round(time.perf_counter() - t_warmup_0, 3)
        except Exception as e:
            err_str = str(e)
            status = ExecutionStatus.SKIPPED_OOM if "out of memory" in err_str.lower() else ExecutionStatus.FAILED_RUNTIME
            return BenchmarkRecord(
                run_id=run_id,
                environment_id=self.environment_id,
                mode=mode,
                execution_status=status,
                model_id=model_id,
                subset=subset,
                audio_id=audio_id,
                audio_duration_s=audio_dur,
                error_message=err_str,
                config=config_snapshot
            )

        # 2. Đo lường lặp lại chính thức kết hợp ResourceMonitor
        inference_times: list[float] = []
        hypotheses: list[str] = []

        try:
            with ResourceMonitor(sample_interval_s=0.05) as monitor:
                for _ in range(self.benchmark_runs):
                    t0 = time.perf_counter()
                    hyp = transcribe_fn(audio_data)
                    inf_time = time.perf_counter() - t0
                    inference_times.append(inf_time)
                    hypotheses.append(hyp)

            resource_data = monitor.get_metrics()
        except Exception as e:
            err_str = str(e)
            status = ExecutionStatus.SKIPPED_OOM if "out of memory" in err_str.lower() else ExecutionStatus.FAILED_RUNTIME
            return BenchmarkRecord(
                run_id=run_id,
                environment_id=self.environment_id,
                mode=mode,
                execution_status=status,
                model_id=model_id,
                subset=subset,
                audio_id=audio_id,
                audio_duration_s=audio_dur,
                error_message=err_str,
                config=config_snapshot
            )

        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
        final_hypothesis = hypotheses[-1] if hypotheses else ""

        # 3. Tính toán độ chính xác (Accuracy Metrics)
        wer = compute_wer(ground_truth, final_hypothesis, normalize=True)
        cer = compute_cer(ground_truth, final_hypothesis, normalize=True)

        term_recall = None
        term_accuracy = None
        if self.technical_terms and subset == "codeswitch_tech":
            term_metrics = calculate_technical_term_metrics(ground_truth, final_hypothesis, self.technical_terms)
            term_recall = term_metrics["technical_term_recall"]
            term_accuracy = term_metrics["technical_term_accuracy"]

        norm_gt = normalize_vietnamese_text(ground_truth)
        norm_hyp = normalize_vietnamese_text(final_hypothesis)

        acc_metrics = AccuracyMetrics(
            normalized_wer=wer,
            normalized_cer=cer,
            technical_term_recall=term_recall,
            technical_term_accuracy=term_accuracy,
            gt_word_count=len(norm_gt.split()),
            hyp_word_count=len(norm_hyp.split())
        )

        # 4. Tính toán hiệu năng (Performance Metrics)
        rtf = calculate_rtf(avg_inference_time, audio_dur)
        throughput = calculate_throughput(avg_inference_time, audio_dur)

        perf_metrics = PerformanceMetrics(
            load_time_s=round(load_time_s, 2),
            warmup_time_s=warmup_time_s,
            asr_inference_time_s=round(avg_inference_time, 3),
            asr_rtf=rtf,
            throughput_x_realtime=throughput,
            total_end_to_end_time_s=round(avg_inference_time, 3)
        )

        res_metrics = ResourceMetrics(
            gpu_memory_source=resource_data["gpu_memory_source"],
            baseline_vram_mb=resource_data["baseline_vram_mb"],
            peak_vram_mb=resource_data["peak_vram_mb"],
            delta_peak_vram_mb=resource_data["delta_peak_vram_mb"],
            baseline_ram_mb=resource_data["baseline_ram_mb"],
            peak_ram_mb=resource_data["peak_ram_mb"],
            delta_peak_ram_mb=resource_data["delta_peak_ram_mb"],
            avg_cpu_percent=resource_data["avg_cpu_percent"],
            peak_cpu_percent=resource_data["peak_cpu_percent"]
        )

        return BenchmarkRecord(
            run_id=run_id,
            environment_id=self.environment_id,
            mode=mode,
            execution_status=ExecutionStatus.COMPLETED,
            model_id=model_id,
            subset=subset,
            audio_id=audio_id,
            audio_duration_s=round(audio_dur, 2),
            accuracy=acc_metrics,
            performance=perf_metrics,
            resource=res_metrics,
            config=config_snapshot
        )

    @staticmethod
    def save_records_to_jsonl(records: list[BenchmarkRecord], jsonl_path: str | Path) -> None:
        """Lưu danh sách bản ghi thực nghiệm vào file JSON Lines (Source of Truth)."""
        path = Path(jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(rec.model_dump_json() + "\n")

    @staticmethod
    def export_records_to_csv(records: list[BenchmarkRecord], csv_path: str | Path) -> None:
        """Xuất danh sách bản ghi ra bảng CSV phẳng phục vụ đọc nhanh và phân tích."""
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for r in records:
            row = {
                "run_id": r.run_id,
                "environment_id": r.environment_id,
                "mode": r.mode,
                "status": r.execution_status.value,
                "model_id": r.model_id,
                "subset": r.subset,
                "audio_id": r.audio_id,
                "audio_duration_s": r.audio_duration_s,
                "wer": r.accuracy.normalized_wer if r.accuracy else None,
                "cer": r.accuracy.normalized_cer if r.accuracy else None,
                "term_recall": r.accuracy.technical_term_recall if r.accuracy else None,
                "term_accuracy": r.accuracy.technical_term_accuracy if r.accuracy else None,
                "inference_time_s": r.performance.asr_inference_time_s if r.performance else None,
                "rtf": r.performance.asr_rtf if r.performance else None,
                "throughput": r.performance.throughput_x_realtime if r.performance else None,
                "delta_vram_mb": r.resource.delta_peak_vram_mb if r.resource else None,
                "delta_ram_mb": r.resource.delta_peak_ram_mb if r.resource else None,
                "avg_cpu_percent": r.resource.avg_cpu_percent if r.resource else None,
                "backend": r.config.backend if r.config else None,
                "precision": r.config.precision if r.config else None
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding="utf-8-sig")
