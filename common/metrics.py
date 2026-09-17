"""
Module tính toán các chỉ số đánh giá (Evaluation Metrics).

Cung cấp:
- Word Error Rate (WER) và Character Error Rate (CER) với chuẩn hóa văn bản.
- Real-Time Factor (RTF) và Throughput (Tốc độ xử lý so với thời gian thực).
- Revision Distance (Chỉ số đo mức độ nhấp nháy, biến động văn bản streaming).
- Technical Term Recall và Technical Term Accuracy cho các thuật ngữ kỹ thuật.
"""

from typing import Sequence, Any
from common.text_normalization import normalize_vietnamese_text, analyze_technical_terms


def levenshtein_distance(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:
    """
    Tính khoảng cách Levenshtein (Edit Distance) tối thiểu giữa hai chuỗi token hoặc ký tự.

    Sử dụng quy hoạch động tối ưu bộ nhớ O(min(N, M)).
    """
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    if len(seq2) == 0:
        return len(seq1)

    previous_row = list(range(len(seq2) + 1))

    for i, c1 in enumerate(seq1):
        current_row = [i + 1] * (len(seq2) + 1)
        for j, c2 in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row

    return previous_row[-1]


def compute_wer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Tính Word Error Rate (WER) giữa chuỗi gốc (reference) và chuỗi dự đoán (hypothesis).

    Công thức: WER = (Substitutions + Deletions + Insertions) / Total Reference Words
    Nếu normalize=True, cả hai chuỗi sẽ được chuẩn hóa Unicode NFC, lowercase, bỏ dấu câu trước.
    """
    if normalize:
        reference = normalize_vietnamese_text(reference)
        hypothesis = normalize_vietnamese_text(hypothesis)

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    edit_dist = levenshtein_distance(ref_words, hyp_words)
    return round(edit_dist / len(ref_words), 4)


def compute_cer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Tính Character Error Rate (CER) ở cấp độ ký tự (không tính khoảng trắng).

    Rất quan trọng cho tiếng Việt để đánh giá độ chính xác của dấu thanh và phụ âm cuối.
    """
    if normalize:
        reference = normalize_vietnamese_text(reference)
        hypothesis = normalize_vietnamese_text(hypothesis)

    ref_chars = [c for c in reference if not c.isspace()]
    hyp_chars = [c for c in hypothesis if not c.isspace()]

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    edit_dist = levenshtein_distance(ref_chars, hyp_chars)
    return round(edit_dist / len(ref_chars), 4)


def calculate_rtf(inference_time: float, audio_duration: float) -> float:
    """
    Tính Real-Time Factor (RTF) của một lượt suy luận ASR.

    RTF = thời gian suy luận / thời lượng âm thanh.
    Lưu ý: RTF < 1.0 chỉ cho biết tốc độ xử lý nhanh hơn thời lượng phát audio,
    hoàn toàn không đồng nghĩa với việc hệ thống có khả năng streaming thực tế.
    """
    if audio_duration <= 0:
        return 0.0
    return round(inference_time / audio_duration, 4)


def calculate_throughput(inference_time: float, audio_duration: float) -> float:
    """
    Tính thông lượng xử lý (Throughput) gấp bao nhiêu lần thời gian thực (x Real-time).

    Throughput = thời lượng âm thanh / thời gian suy luận = 1 / RTF.
    """
    if inference_time <= 0:
        return 0.0
    return round(audio_duration / inference_time, 2)


def calculate_revision_distance(previous_text: str, current_text: str) -> int:
    """
    Tính độ biến động (Revision Distance) giữa 2 partial transcript liên tiếp trong streaming.

    Đo lường mức độ văn bản bị sửa đổi, đảo ngược hoặc thay thế, phản ánh độ nhấp nháy UI.
    """
    prev_words = normalize_vietnamese_text(previous_text).split()
    curr_words = normalize_vietnamese_text(current_text).split()
    return levenshtein_distance(prev_words, curr_words)


def calculate_technical_term_metrics(
    reference: str,
    hypothesis: str,
    terms_list: list[str]
) -> dict:
    """
    Đo lường các chỉ số chuyên biệt cho thuật ngữ kỹ thuật (Code-Switching).

    Trả về dictionary gồm:
    - technical_term_recall: Tỷ lệ thuật ngữ trong ground truth được mô hình bắt trúng.
    - technical_term_accuracy: Tỷ lệ thuật ngữ nhận dạng đúng trên tổng số lần xuất hiện.
    """
    analysis = analyze_technical_terms(reference, hypothesis, terms_list)
    return {
        "technical_term_recall": analysis["term_recall"],
        "technical_term_accuracy": analysis["term_accuracy"],
        "term_total_in_gt": analysis["total_in_gt"],
        "term_correct": analysis["correct"],
        "term_substitution": analysis["substitution"],
        "term_deletion": analysis["deletion"],
        "term_details": analysis["details"]
    }
