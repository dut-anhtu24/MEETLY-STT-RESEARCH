#!/usr/bin/env python3
"""
CLI điều phối thực nghiệm benchmark STT Meetly trên Server GPU / Máy trạm.

Cung cấp khả năng chạy headless không phụ thuộc vào giao diện Jupyter:
- Mode screening: Sàng lọc ứng viên mô hình (Whisper Small, Turbo, Large v3, PhoWhisper)
- Mode offline: Đo kiểm pipeline ngoại tuyến (Baseline vs faster-whisper FP16/INT8 + VAD)
- Mode streaming: Đo kiểm mô phỏng trực tuyến (LocalAgreement, TTFP, Revision Count, Latency)
- Mode all: Chạy toàn bộ chuỗi thực nghiệm tuần tự

Hỗ trợ:
- Tự động nhận diện phần cứng GPU NVIDIA / CPU
- Nhận diện biến môi trường CUDA_VISIBLE_DEVICES và tham số --gpu-id
- Cấu hình thư mục cache Hugging Face qua HF_HOME
- Tự động bắt lỗi OOM và Unsupported backend mà không làm crash cả campaign
- Xuất kết quả ra JSON Lines (Source of Truth), bảng CSV và experiment_environment.json
"""

import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from typing import Any, Callable
import yaml
import pandas as pd
import numpy as np
import torch

# Thiết lập đường dẫn thư mục gốc dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Đảm bảo mã hóa UTF-8 an toàn trên mọi hệ điều hành (Windows/Linux)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from common.audio import load_audio, get_audio_duration
from common.model_registry import ModelRegistry, ModelMetadata
from common.benchmark_runner import BenchmarkRunner
from common.offline_engine import OfflinePipelineEngine
from common.streaming_engine import StreamingSimulationEngine
from common.result_schema import (
    EnvironmentFingerprint,
    BenchmarkRecord,
    ExecutionStatus,
    ConfigSnapshot,
    AccuracyMetrics,
    PerformanceMetrics,
    ResourceMetrics
)


def resolve_device(requested_device: str = "auto", gpu_id: int | None = None) -> tuple[str, int | None]:
    """
    Xác định thiết bị tính toán (device) và GPU index phù hợp.
    
    :param requested_device: 'auto', 'cuda', hoặc 'cpu'
    :param gpu_id: ID của GPU mong muốn (0, 1, ...)
    :return: Tuple (device_str, active_gpu_id)
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if requested_device == "cpu":
        return "cpu", None

    cuda_available = torch.cuda.is_available()
    if requested_device == "cuda" and not cuda_available:
        print("⚠️ Cảnh báo: Yêu cầu CUDA nhưng PyTorch không phát hiện GPU khả dụng. Chuyển về CPU.")
        return "cpu", None

    if cuda_available:
        active_id = torch.cuda.current_device() if gpu_id is None else 0
        device_str = f"cuda:{active_id}"
        return device_str, active_id

    return "cpu", None


def get_model_transcriber(
    metadata: ModelMetadata,
    backend: str,
    device: str,
    precision: str,
    hf_cache_dir: str | None = None,
    mock_mode: bool = False
) -> tuple[Callable[[np.ndarray], str], float]:
    """
    Nạp mô hình và tạo hàm phiên âm callable (transcribe_fn).
    Đo đạc thời gian nạp trọng số riêng biệt (không tính vào thời gian suy luận benchmark).

    :return: Tuple (transcribe_fn, load_time_s)
    """
    if mock_mode:
        t0 = time.perf_counter()
        time.sleep(0.05)
        load_time = time.perf_counter() - t0
        def mock_transcribe(audio_array: np.ndarray) -> str:
            time.sleep(0.02)
            return "chúng tôi thử nghiệm mô hình phiên âm tiếng việt trong cuộc họp"
        return mock_transcribe, load_time

    # 1. Kiểm tra tính tương thích backend
    if backend not in metadata.supported_backends:
        raise NotImplementedError(
            f"Mô hình '{metadata.model_id}' không hỗ trợ backend '{backend}'. "
            f"Danh sách backend hỗ trợ: {metadata.supported_backends}"
        )

    if metadata.conversion_required and backend == "faster-whisper":
        raise NotImplementedError(
            f"Mô hình '{metadata.model_id}' yêu cầu convert trước khi chạy trên faster-whisper "
            f"(chưa có checkpoint ct2 chính thức)."
        )

    t0 = time.perf_counter()

    # 2. Khởi tạo engine theo backend
    if backend == "faster-whisper":
        from faster_whisper import WhisperModel
        fw_device = "cuda" if "cuda" in device else "cpu"
        compute_type = "float16" if (fw_device == "cuda" and precision in ["float16", "fp16"]) else ("int8" if "int8" in precision else "float32")
        
        # Nạp mô hình Whisper CTranslate2
        model = WhisperModel(
            metadata.model_id,
            device=fw_device,
            compute_type=compute_type,
            download_root=hf_cache_dir
        )
        load_time = time.perf_counter() - t0

        def faster_whisper_transcribe(audio_array: np.ndarray) -> str:
            # faster-whisper chấp nhận mảng numpy float32 16kHz trực tiếp
            segments, _ = model.transcribe(
                audio_array,
                language="vi",
                beam_size=5,
                vad_filter=False
            )
            return " ".join([seg.text.strip() for seg in segments])

        return faster_whisper_transcribe, load_time

    elif backend == "transformers":
        from transformers import pipeline
        torch_dtype = torch.float16 if ("cuda" in device and precision in ["float16", "fp16"]) else torch.float32
        device_arg = 0 if "cuda" in device else -1

        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=metadata.model_id,
            device=device_arg,
            torch_dtype=torch_dtype,
            model_kwargs={"cache_dir": hf_cache_dir} if hf_cache_dir else None
        )
        load_time = time.perf_counter() - t0

        def transformers_transcribe(audio_array: np.ndarray) -> str:
            res = asr_pipeline(audio_array, generate_kwargs={"language": "vi", "task": "transcribe"})
            return res.get("text", "").strip()

        return transformers_transcribe, load_time

    else:
        raise ValueError(f"Backend không xác định: {backend}")


def run_screening_campaign(
    registry: ModelRegistry,
    bench_cfg: dict,
    metadata_df: pd.DataFrame,
    env_fingerprint: EnvironmentFingerprint,
    device: str,
    mock_mode: bool = False
) -> list[BenchmarkRecord]:
    """Thực thi chiến dịch sàng lọc mô hình (Checkpoint 2)."""
    print("\n" + "=" * 80)
    print("🚀 BẮT ĐẦU SÀNG LỌC MÔ HÌNH (MODEL SCREENING BENCHMARK)")
    print("=" * 80)

    tech_terms = bench_cfg.get("technical_terms", [])
    runner = BenchmarkRunner(
        environment_id=env_fingerprint.environment_id,
        warmup_runs=bench_cfg["protocol"].get("warmup_runs", 1),
        benchmark_runs=bench_cfg["protocol"].get("benchmark_runs", 2),
        technical_terms=tech_terms
    )

    candidates = registry.list_candidates(include_smoke_test=False)
    records: list[BenchmarkRecord] = []
    hf_cache = os.environ.get("HF_HOME", None)

    for cand in candidates:
        backend = cand.preferred_backend
        precision = cand.default_precision
        print(f"\n👉 Đang thẩm định: [{cand.name}] ({cand.model_id}) | Backend: {backend} | Precision: {precision}")

        # Nạp mô hình an toàn với kiểm soát OOM và Unsupported
        try:
            transcribe_fn, load_time_s = get_model_transcriber(
                metadata=cand,
                backend=backend,
                device=device,
                precision=precision,
                hf_cache_dir=hf_cache,
                mock_mode=mock_mode
            )
            print(f"   ✓ Nạp mô hình thành công (Load time: {load_time_s:.2f}s)")
        except NotImplementedError as e:
            print(f"   ⏭️ Bỏ qua do backend không hỗ trợ: {e}")
            rec = BenchmarkRecord(
                run_id=f"scr_{cand.name.lower().replace(' ', '_')}_skipped",
                environment_id=env_fingerprint.environment_id,
                mode="screening",
                execution_status=ExecutionStatus.SKIPPED_UNSUPPORTED,
                model_id=cand.model_id,
                subset="all",
                audio_id="none",
                audio_duration_s=0.0,
                error_message=str(e),
                config=ConfigSnapshot(backend=backend, precision=precision, device=device)
            )
            records.append(rec)
            continue
        except Exception as e:
            err_msg = str(e)
            is_oom = "out of memory" in err_msg.lower() or "cuda oom" in err_msg.lower()
            status = ExecutionStatus.SKIPPED_OOM if is_oom else ExecutionStatus.FAILED_RUNTIME
            icon = "💥 OOM" if is_oom else "❌ Lỗi Runtime"
            print(f"   {icon}: {err_msg[:120]}")
            rec = BenchmarkRecord(
                run_id=f"scr_{cand.name.lower().replace(' ', '_')}_failed",
                environment_id=env_fingerprint.environment_id,
                mode="screening",
                execution_status=status,
                model_id=cand.model_id,
                subset="all",
                audio_id="none",
                audio_duration_s=0.0,
                error_message=err_msg,
                config=ConfigSnapshot(backend=backend, precision=precision, device=device)
            )
            records.append(rec)
            continue

        # Chạy trên các file mẫu kiểm thử trong dataset (rút gọn 1 file nếu là mock mode)
        sample_rows = metadata_df.head(1) if mock_mode else metadata_df
        for _, row in sample_rows.iterrows():
            audio_path = PROJECT_ROOT / row["file_path"]
            gt_path = PROJECT_ROOT / row["ground_truth_path"]
            ground_truth = gt_path.read_text(encoding="utf-8").strip() if gt_path.exists() else ""
            run_id = f"scr_{cand.name.lower().replace(' ', '_')}_{row['audio_id']}"

            config_snap = ConfigSnapshot(
                backend=backend,
                precision=precision,
                device=device,
                batch_size=1
            )

            rec = runner.run_single_experiment(
                run_id=run_id,
                mode="screening",
                model_id=cand.model_id,
                subset=row["subset"],
                audio_id=row["audio_id"],
                audio_path=audio_path,
                ground_truth=ground_truth,
                transcribe_fn=transcribe_fn,
                load_time_s=load_time_s,
                config_snapshot=config_snap
            )
            records.append(rec)
            status_symbol = "✓" if rec.execution_status == ExecutionStatus.COMPLETED else "✗"
            wer_display = f"{rec.accuracy.normalized_wer * 100:.1f}%" if rec.accuracy else "N/A"
            rtf_display = f"{rec.performance.asr_rtf:.3f}" if rec.performance else "N/A"
            print(f"   [{status_symbol}] Audio: {row['audio_id']:<15} | WER: {wer_display:<7} | RTF: {rtf_display:<7} | Status: {rec.execution_status.value}", flush=True)

        # Dọn dẹp VRAM sau mỗi model
        if "cuda" in device and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Lưu kết quả
    jsonl_path = PROJECT_ROOT / "results" / "model_screening.jsonl"
    csv_path = PROJECT_ROOT / "results" / "model_screening.csv"
    runner.save_records_to_jsonl(records, jsonl_path)
    runner.export_records_to_csv(records, csv_path)
    print(f"\n💾 Đã lưu kết quả sàng lọc: {csv_path} ({len(records)} bản ghi)", flush=True)
    return records


def run_offline_campaign(
    offline_cfg: dict,
    metadata_df: pd.DataFrame,
    env_fingerprint: EnvironmentFingerprint,
    device: str,
    mock_mode: bool = False
) -> list[BenchmarkRecord]:
    """Thực thi đo kiểm các biến thể Pipeline Ngoại Tuyến (Checkpoint 3)."""
    print("\n" + "=" * 80, flush=True)
    print("🎥 BẮT ĐẦU ĐO KIỂM PIPELINE NGOẠI TUYẾN (OFFLINE BENCHMARK)", flush=True)
    print("=" * 80, flush=True)

    pipelines_dict = offline_cfg.get("pipelines", {})
    records: list[BenchmarkRecord] = []
    hf_cache = os.environ.get("HF_HOME", None)

    # Chọn model tiêu chuẩn cho offline benchmark (Whisper Large v3 Turbo)
    target_model_id = "openai/whisper-large-v3-turbo"
    dummy_meta = ModelMetadata(
        name="Whisper Large v3 Turbo",
        model_id=target_model_id,
        family="Whisper-Turbo",
        architecture="encoder_decoder",
        supported_backends=["transformers", "faster-whisper"],
        preferred_backend="faster-whisper"
    )

    # Lọc các file dài hoặc meeting để đo offline (mock mode dùng 1 file ngắn để chạy nhanh)
    if mock_mode:
        target_samples = metadata_df[metadata_df["subset"] == "clean_vi"].head(1)
    else:
        target_samples = metadata_df[metadata_df["subset"].isin(["clean_vi", "meeting_natural", "longform_meeting"])]

    for pipe_key, p_info in pipelines_dict.items():
        backend = p_info.get("backend", "faster-whisper")
        precision = p_info.get("precision", "float16")
        use_vad = False if mock_mode else p_info.get("use_vad", False)
        vad_provider = p_info.get("vad_provider", "silero")
        vad_threshold = p_info.get("vad_threshold", 0.5)

        print(f"\n👉 Pipeline: [{pipe_key}] | Backend: {backend} | Precision: {precision} | VAD: {use_vad}", flush=True)

        try:
            transcribe_fn, load_time_s = get_model_transcriber(
                metadata=dummy_meta,
                backend=backend,
                device=device,
                precision=precision,
                hf_cache_dir=hf_cache,
                mock_mode=mock_mode
            )
        except Exception as e:
            print(f"   ❌ Lỗi khởi tạo pipeline: {e}")
            continue

        engine = OfflinePipelineEngine(
            environment_id=env_fingerprint.environment_id,
            vad_provider=vad_provider if use_vad else "none",
            vad_threshold=vad_threshold
        )

        for _, row in target_samples.iterrows():
            audio_path = PROJECT_ROOT / row["file_path"]
            gt_path = PROJECT_ROOT / row["ground_truth_path"]
            ground_truth = gt_path.read_text(encoding="utf-8").strip() if gt_path.exists() else ""

            _, rec = engine.process_offline(
                input_path=audio_path,
                model_id=target_model_id,
                backend=backend,
                precision=precision,
                transcribe_fn=transcribe_fn,
                use_vad=use_vad,
                is_video=False,
                ground_truth=ground_truth,
                session_id=f"session_{row['audio_id']}"
            )
            rec.run_id = f"off_{pipe_key}_{row['audio_id']}"
            records.append(rec)

            wer_val = f"{rec.accuracy.normalized_wer * 100:.1f}%" if rec.accuracy else "N/A"
            rtf_val = f"{rec.performance.asr_rtf:.3f}" if rec.performance else "N/A"
            vram_val = f"{rec.resource.delta_peak_vram_mb:.1f}MB" if rec.resource else "N/A"
            print(f"   ✓ File: {row['audio_id']:<16} | WER: {wer_val:<7} | RTF: {rtf_val:<7} | Delta VRAM: {vram_val}", flush=True)

        if "cuda" in device and torch.cuda.is_available():
            torch.cuda.empty_cache()

    jsonl_path = PROJECT_ROOT / "results" / "offline_pipeline_benchmark.jsonl"
    csv_path = PROJECT_ROOT / "results" / "offline_pipeline_benchmark.csv"
    BenchmarkRunner.save_records_to_jsonl(records, jsonl_path)
    BenchmarkRunner.export_records_to_csv(records, csv_path)
    print(f"\n💾 Đã lưu kết quả offline: {csv_path} ({len(records)} bản ghi)", flush=True)
    return records


def run_streaming_campaign(
    streaming_cfg: dict,
    metadata_df: pd.DataFrame,
    env_fingerprint: EnvironmentFingerprint,
    device: str,
    mock_mode: bool = False
) -> list[BenchmarkRecord]:
    """Thực thi đo kiểm mô phỏng trực tuyến Streaming (Checkpoint 4)."""
    print("\n" + "=" * 80, flush=True)
    print("🎙️ BẮT ĐẦU MÔ PHỎNG TRỰC TUYẾN STREAMING (STREAMING BENCHMARK)", flush=True)
    print("=" * 80, flush=True)

    pipelines_dict = streaming_cfg.get("pipelines", {})
    records: list[BenchmarkRecord] = []
    events_jsonl_path = PROJECT_ROOT / "results" / "stream_events.jsonl"
    hf_cache = os.environ.get("HF_HOME", None)

    # Chọn mẫu ngắn cho streaming (1 mẫu nếu mock mode để kiểm thử nhanh)
    if mock_mode:
        stream_samples = metadata_df[metadata_df["subset"] == "clean_vi"].head(1)
    else:
        stream_samples = metadata_df[metadata_df["subset"].isin(["clean_vi", "meeting_natural"])].head(2)

    for pipe_key, p_info in pipelines_dict.items():
        model_id = p_info.get("model_id", "openai/whisper-small")
        backend = p_info.get("backend", "faster-whisper")
        precision = p_info.get("precision", "float16")
        step_ms = p_info.get("step_duration_ms", 500)
        ctx_s = p_info.get("context_window_s", 3.0)
        la_threshold = p_info.get("local_agreement_k", 2)

        print(f"\n👉 Pipeline: [{pipe_key}] | Step: {step_ms}ms | Context: {ctx_s}s | LA-k: {la_threshold}", flush=True)

        dummy_meta = ModelMetadata(
            name="Stream Model",
            model_id=model_id,
            family="Whisper",
            architecture="encoder_decoder",
            supported_backends=["faster-whisper", "transformers"],
            preferred_backend=backend
        )

        try:
            transcribe_fn, _ = get_model_transcriber(
                metadata=dummy_meta,
                backend=backend,
                device=device,
                precision=precision,
                hf_cache_dir=hf_cache,
                mock_mode=mock_mode
            )
        except Exception as e:
            print(f"   ❌ Lỗi nạp mô hình streaming: {e}")
            continue

        engine = StreamingSimulationEngine(
            environment_id=env_fingerprint.environment_id,
            frame_duration_ms=100,
            step_duration_ms=step_ms,
            context_window_s=ctx_s,
            min_common_prefix_tokens=la_threshold
        )

        for _, row in stream_samples.iterrows():
            audio_path = PROJECT_ROOT / row["file_path"]
            gt_path = PROJECT_ROOT / row["ground_truth_path"]
            ground_truth = gt_path.read_text(encoding="utf-8").strip() if gt_path.exists() else ""

            rec, events = engine.run_streaming_simulation(
                audio_path=audio_path,
                model_id=model_id,
                backend=backend,
                precision=precision,
                incremental_transcribe_fn=transcribe_fn,
                simulation_mode="virtual_clock",
                ground_truth=ground_truth,
                session_id=f"stream_{pipe_key}_{row['audio_id']}"
            )
            rec.run_id = f"str_{pipe_key}_{row['audio_id']}"
            records.append(rec)

            # Lưu các sự kiện stream chi tiết
            with open(events_jsonl_path, "a", encoding="utf-8") as f:
                for ev in events:
                    f.write(ev.model_dump_json() + "\n")

            ttfp_val = f"{rec.streaming.first_partial_latency_s:.3f}s" if (rec.streaming and rec.streaming.first_partial_latency_s is not None) else "N/A"
            p95_val = f"{rec.streaming.p95_processing_latency_s:.3f}s" if (rec.streaming and rec.streaming.p95_processing_latency_s is not None) else "N/A"
            rev_val = f"{rec.streaming.revision_count}" if rec.streaming else "N/A"
            wer_val = f"{rec.accuracy.normalized_wer * 100:.1f}%" if rec.accuracy else "N/A"
            print(f"   ✓ File: {row['audio_id']:<15} | TTFP: {ttfp_val:<8} | P95: {p95_val:<8} | Revisions: {rev_val:<4} | WER: {wer_val}", flush=True)

        if "cuda" in device and torch.cuda.is_available():
            torch.cuda.empty_cache()

    jsonl_path = PROJECT_ROOT / "results" / "streaming_pipeline_benchmark.jsonl"
    csv_path = PROJECT_ROOT / "results" / "streaming_pipeline_benchmark.csv"
    BenchmarkRunner.save_records_to_jsonl(records, jsonl_path)
    BenchmarkRunner.export_records_to_csv(records, csv_path)
    print(f"\n💾 Đã lưu kết quả streaming: {csv_path} ({len(records)} bản ghi)")
    return records


def main():
    parser = argparse.ArgumentParser(description="Chương trình điều phối thực nghiệm benchmark STT Meetly.")
    parser.add_argument(
        "--mode",
        choices=["screening", "offline", "streaming", "all"],
        default="all",
        help="Chế độ thực thi benchmark (screening, offline, streaming, all)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Thiết bị thực thi tính toán (mặc định: auto - tự nhận diện GPU NVIDIA)"
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Chỉ định GPU index cụ thể (tương đương CUDA_VISIBLE_DEVICES)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Chạy ở chế độ mô phỏng suy luận để kiểm thử tính toàn vẹn của pipeline trước khi tải weights lớn"
    )
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default=str(PROJECT_ROOT / "data" / "metadata.csv"),
        help="Đường dẫn file metadata CSV của dataset"
    )

    args = parser.parse_args()

    # 1. Xác thực thiết bị tính toán
    active_device, active_gpu_id = resolve_device(args.device, args.gpu_id)
    print(f"🖥️ Thiết bị suy luận được kích hoạt: {active_device} (GPU ID: {active_gpu_id})")

    # 2. Thu thập và lưu vết Environment Fingerprint
    env_fingerprint = EnvironmentFingerprint.capture(
        gpu_id=active_gpu_id,
        environment_id=f"dutai_gpu_env_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    env_file = PROJECT_ROOT / "results" / "experiment_environment.json"
    env_fingerprint.save_to_json(env_file)
    print(f"📋 Đã lưu dấu vân tay môi trường: {env_file}")
    print(f"   • OS: {env_fingerprint.os}")
    print(f"   • Python: {env_fingerprint.python_version} | PyTorch: {env_fingerprint.pytorch_version}")
    print(f"   • GPU: {env_fingerprint.gpu_name} (VRAM: {env_fingerprint.gpu_total_vram_mb} MB) | CUDA: {env_fingerprint.cuda_version}")
    print(f"   • HF Cache (HF_HOME): {env_fingerprint.hf_home or 'Mặc định'}")

    # 3. Đọc dữ liệu dataset và cấu hình
    metadata_path = Path(args.dataset_csv)
    if not metadata_path.exists():
        print(f"❌ Không tìm thấy dataset metadata tại {metadata_path}. Đang tạo dữ liệu mẫu...")
        from data.prepare_dataset import main as prep_main
        prep_main()

    metadata_df = pd.read_csv(metadata_path)
    models_cfg_path = PROJECT_ROOT / "configs" / "models.yaml"
    bench_cfg_path = PROJECT_ROOT / "configs" / "benchmark.yaml"
    offline_cfg_path = PROJECT_ROOT / "configs" / "offline_pipelines.yaml"
    streaming_cfg_path = PROJECT_ROOT / "configs" / "streaming_pipelines.yaml"

    with open(bench_cfg_path, "r", encoding="utf-8") as f:
        bench_cfg = yaml.safe_load(f)
    with open(offline_cfg_path, "r", encoding="utf-8") as f:
        offline_cfg = yaml.safe_load(f)
    with open(streaming_cfg_path, "r", encoding="utf-8") as f:
        streaming_cfg = yaml.safe_load(f)

    registry = ModelRegistry(models_cfg_path)

    # 4. Điều phối thực thi theo mode
    t_start = time.perf_counter()

    if args.mode in ["screening", "all"]:
        run_screening_campaign(
            registry=registry,
            bench_cfg=bench_cfg,
            metadata_df=metadata_df,
            env_fingerprint=env_fingerprint,
            device=active_device,
            mock_mode=args.mock
        )

    if args.mode in ["offline", "all"]:
        run_offline_campaign(
            offline_cfg=offline_cfg,
            metadata_df=metadata_df,
            env_fingerprint=env_fingerprint,
            device=active_device,
            mock_mode=args.mock
        )

    if args.mode in ["streaming", "all"]:
        run_streaming_campaign(
            streaming_cfg=streaming_cfg,
            metadata_df=metadata_df,
            env_fingerprint=env_fingerprint,
            device=active_device,
            mock_mode=args.mock
        )

    elapsed_s = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"🎉 HOÀN TẤT THỰC NGHIỆM ({args.mode.upper()}) TRONG {elapsed_s:.2f}s!")
    print(f"📁 Toàn bộ kết quả đã được ghi nhận tại thư mục: {PROJECT_ROOT / 'results'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
