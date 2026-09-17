"""
Script đo kiểm thực nghiệm VIVOS Test Set cho Meetly STT Research.

Đo đạc và so sánh trực tiếp:
- Wav2Vec2 Vietnamese 250h (nguyenvulebinh/wav2vec2-base-vietnamese-250h)
- Whisper Small (faster-whisper INT8 / FP32)
- PhoWhisper Small (vinai/PhoWhisper-small - nếu đã tải xong)

Các chỉ số tính toán chuẩn hóa:
- Normalized WER & CER (theo chuẩn NFC tiếng Việt Lớp 1 của Meetly)
- Latency (s), RTF (Real-time factor), Throughput (x Real-time)
"""

import sys
import time
import csv
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.text_normalization import normalize_vietnamese_text
from common.metrics import compute_wer, compute_cer, calculate_rtf, calculate_throughput

def run_vivos_eval():
    metadata_path = PROJECT_ROOT / "data" / "vivos_test_sample.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file metadata: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)
    print(f"📊 Đã nạp {len(df_meta)} mẫu audio từ {metadata_path}")
    total_audio_s = df_meta["duration_s"].sum()
    print(f"⏱️ Tổng thời lượng audio đánh giá: {total_audio_s:.1f} giây across {len(df_meta)} speakers\n")

    results = []

    # =========================================================================
    # 1. EVALUATE: Wav2Vec2 Vietnamese 250h (nguyenvulebinh)
    # =========================================================================
    print("=" * 80)
    print("🚀 [1/3] ĐO KIỂM: Wav2Vec2 Vietnamese 250h (nguyenvulebinh/wav2vec2-base-vietnamese-250h)")
    print("=" * 80)
    
    try:
        from transformers import AutoProcessor, AutoModelForCTC
        t0_load = time.perf_counter()
        w2v_proc = AutoProcessor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
        w2v_model = AutoModelForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
        w2v_load_time = time.perf_counter() - t0_load
        print(f"✓ Nạp mô hình thành công trong {w2v_load_time:.2f}s")

        for idx, row in df_meta.iterrows():
            audio_path = PROJECT_ROOT / row["file_path"]
            gt_path = PROJECT_ROOT / row["ground_truth_path"]
            gt_text = gt_path.read_text(encoding="utf-8").strip()
            speech, sr = sf.read(str(audio_path))

            t0 = time.perf_counter()
            inputs = w2v_proc(speech, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = w2v_model(inputs.input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred_text = w2v_proc.batch_decode(pred_ids)[0]
            infer_time = time.perf_counter() - t0

            # Tính metrics
            norm_ref = normalize_vietnamese_text(gt_text)
            norm_hyp = normalize_vietnamese_text(pred_text)
            wer = compute_wer(norm_ref, norm_hyp)
            cer = compute_cer(norm_ref, norm_hyp)
            rtf = calculate_rtf(infer_time, row["duration_s"])

            results.append({
                "model": "Wav2Vec2-vi-250h",
                "backend": "transformers",
                "precision": "float32",
                "audio_id": row["audio_id"],
                "duration_s": row["duration_s"],
                "inference_time_s": infer_time,
                "rtf": rtf,
                "wer": wer,
                "cer": cer,
                "ref_norm": norm_ref,
                "hyp_norm": norm_hyp
            })
            print(f"[{idx+1:02d}/{len(df_meta)}] {row['audio_id']} | Dur: {row['duration_s']:.1f}s | RTF: {rtf:.3f} | WER: {wer*100:.1f}%")
            print(f"   REF: {norm_ref}")
            print(f"   HYP: {norm_hyp}\n")

    except Exception as e:
        print(f"❌ Lỗi khi chạy Wav2Vec2: {e}")

    # =========================================================================
    # 2. EVALUATE: Whisper Small (faster-whisper INT8)
    # =========================================================================
    print("=" * 80)
    print("🚀 [2/3] ĐO KIỂM: Whisper Small (faster-whisper INT8 CPU)")
    print("=" * 80)
    
    try:
        from faster_whisper import WhisperModel
        t0_load = time.perf_counter()
        fw_model = WhisperModel("small", device="cpu", compute_type="int8")
        fw_load_time = time.perf_counter() - t0_load
        print(f"✓ Nạp mô hình thành công trong {fw_load_time:.2f}s")

        for idx, row in df_meta.iterrows():
            audio_path = PROJECT_ROOT / row["file_path"]
            gt_path = PROJECT_ROOT / row["ground_truth_path"]
            gt_text = gt_path.read_text(encoding="utf-8").strip()
            speech, sr = sf.read(str(audio_path))

            t0 = time.perf_counter()
            segments, _ = fw_model.transcribe(speech, language="vi", beam_size=5)
            pred_text = " ".join(s.text.strip() for s in segments)
            infer_time = time.perf_counter() - t0

            # Tính metrics
            norm_ref = normalize_vietnamese_text(gt_text)
            norm_hyp = normalize_vietnamese_text(pred_text)
            wer = compute_wer(norm_ref, norm_hyp)
            cer = compute_cer(norm_ref, norm_hyp)
            rtf = calculate_rtf(infer_time, row["duration_s"])

            results.append({
                "model": "Whisper-Small",
                "backend": "faster-whisper",
                "precision": "int8",
                "audio_id": row["audio_id"],
                "duration_s": row["duration_s"],
                "inference_time_s": infer_time,
                "rtf": rtf,
                "wer": wer,
                "cer": cer,
                "ref_norm": norm_ref,
                "hyp_norm": norm_hyp
            })
            print(f"[{idx+1:02d}/{len(df_meta)}] {row['audio_id']} | Dur: {row['duration_s']:.1f}s | RTF: {rtf:.3f} | WER: {wer*100:.1f}%")
            print(f"   REF: {norm_ref}")
            print(f"   HYP: {norm_hyp}\n")

    except Exception as e:
        print(f"❌ Lỗi khi chạy Whisper Small: {e}")

    # =========================================================================
    # 3. EVALUATE: PhoWhisper Small (Nếu đã tải hoàn tất)
    # =========================================================================
    print("=" * 80)
    print("🚀 [3/3] ĐO KIỂM: PhoWhisper Small (vinai/PhoWhisper-small)")
    print("=" * 80)
    try:
        from transformers import pipeline
        t0_load = time.perf_counter()
        pho_pipeline = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small", device=-1)
        pho_load_time = time.perf_counter() - t0_load
        print(f"✓ Nạp mô hình PhoWhisper thành công trong {pho_load_time:.2f}s")

        for idx, row in df_meta.iterrows():
            audio_path = PROJECT_ROOT / row["file_path"]
            gt_path = PROJECT_ROOT / row["ground_truth_path"]
            gt_text = gt_path.read_text(encoding="utf-8").strip()
            speech, sr = sf.read(str(audio_path))

            t0 = time.perf_counter()
            res = pho_pipeline(speech, generate_kwargs={"language": "vi", "task": "transcribe"})
            pred_text = res.get("text", "").strip()
            infer_time = time.perf_counter() - t0

            # Tính metrics
            norm_ref = normalize_vietnamese_text(gt_text)
            norm_hyp = normalize_vietnamese_text(pred_text)
            wer = compute_wer(norm_ref, norm_hyp)
            cer = compute_cer(norm_ref, norm_hyp)
            rtf = calculate_rtf(infer_time, row["duration_s"])

            results.append({
                "model": "PhoWhisper-Small",
                "backend": "transformers",
                "precision": "float32",
                "audio_id": row["audio_id"],
                "duration_s": row["duration_s"],
                "inference_time_s": infer_time,
                "rtf": rtf,
                "wer": wer,
                "cer": cer,
                "ref_norm": norm_ref,
                "hyp_norm": norm_hyp
            })
            print(f"[{idx+1:02d}/{len(df_meta)}] {row['audio_id']} | Dur: {row['duration_s']:.1f}s | RTF: {rtf:.3f} | WER: {wer*100:.1f}%")
            print(f"   REF: {norm_ref}")
            print(f"   HYP: {norm_hyp}\n")

    except Exception as e:
        print(f"⚠️ PhoWhisper Small chưa khả dụng hoặc gặp lỗi: {e}")

    # =========================================================================
    # TỔNG HỢP VÀ XUẤT BẢNG SO SÁNH
    # =========================================================================
    if not results:
        print("❌ Không có kết quả nào được tạo ra.")
        return

    df_res = pd.DataFrame(results)
    out_csv = PROJECT_ROOT / "results" / "vivos_test_benchmark_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n💾 Đã lưu kết quả chi tiết: {out_csv}")

    # Nhóm theo model
    summary = []
    for model_name, grp in df_res.groupby("model"):
        mean_wer = grp["wer"].mean()
        mean_cer = grp["cer"].mean()
        mean_rtf = grp["rtf"].mean()
        throughput = 1.0 / mean_rtf if mean_rtf > 0 else 0
        total_time = grp["inference_time_s"].sum()
        total_audio = grp["duration_s"].sum()
        summary.append({
            "Mô Hình": model_name,
            "Backend": grp["backend"].iloc[0],
            "Precision": grp["precision"].iloc[0],
            "WER Trung Bình": f"{mean_wer * 100:.2f}%",
            "CER Trung Bình": f"{mean_cer * 100:.2f}%",
            "RTF Trung Bình": f"{mean_rtf:.3f}",
            "Thông Lượng (Throughput)": f"{throughput:.1f}x",
            "Tổng thời gian suy luận (s)": f"{total_time:.2f}s",
            "Tổng thời lượng audio (s)": f"{total_audio:.1f}s",
            "Số mẫu đo": len(grp)
        })

    df_summary = pd.DataFrame(summary)
    summary_csv = PROJECT_ROOT / "results" / "vivos_test_summary.csv"
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8")
    print(f"💾 Đã lưu bảng tổng hợp: {summary_csv}\n")

    print("=" * 80)
    print("🏆 BẢNG TỔNG HỢP ĐỐI ĐẦU TRÊN TẬP VIVOS TEST")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    run_vivos_eval()
