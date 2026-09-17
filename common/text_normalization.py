"""
Module chuẩn hóa văn bản tiếng Việt 2 lớp (Vietnamese Text Normalization & Technical Term Analysis).

Cung cấp:
- Lớp 1: Chuẩn hóa NFC, chuyển chữ thường, loại bỏ dấu câu và chuẩn hóa khoảng trắng cho bài toán ASR chuẩn.
- Lớp 2: Bộ trích xuất và phân tích lỗi thuật ngữ kỹ thuật tiếng Anh cho tập VI-EN Code-Switching.
"""

import re
import unicodedata
from typing import TypedDict


class TechnicalTermResult(TypedDict):
    total_in_gt: int
    correct: int
    substitution: int
    deletion: int
    term_recall: float
    term_accuracy: float
    details: list[dict]


# Tập hợp các ký tự dấu câu chuẩn và dấu ngoặc, dấu nháy cần loại bỏ
PUNCTUATION_REGEX = re.compile(r"""[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~“”„…«»—–]""")


# Ánh xạ chuẩn hóa vị trí dấu thanh tiếng Việt (chuyển kiểu cũ sang kiểu mới chuẩn hóa)
TONE_POSITION_MAP = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
    "uà": "ùa", "uá": "úa", "uả": "ủa", "uã": "ũa", "uạ": "ụa"
}


def normalize_vietnamese_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt phục vụ tính toán WER/CER tiêu chuẩn.

    Các bước:
    1. Chuẩn hóa dạng biểu diễn ký tự về Unicode NFC (tránh lệch mã giữa 'hoà' và 'hòa').
    2. Đồng nhất vị trí dấu thanh (chuyển dấu kiểu cũ như 'hoà' sang 'hòa').
    3. Chuyển toàn bộ ký tự sang chữ thường (lowercase).
    4. Loại bỏ dấu câu, dấu ngoặc và ký tự định dạng.
    5. Gộp nhiều khoảng trắng liền kề thành 1 khoảng trắng đơn.
    """
    if not text:
        return ""

    # Bước 1: Unicode NFC
    normalized = unicodedata.normalize("NFC", text)

    # Bước 2: Lowercase
    normalized = normalized.lower()

    # Bước 3: Đồng nhất vị trí đặt dấu thanh tiếng Việt
    for old_tone, new_tone in TONE_POSITION_MAP.items():
        if old_tone in normalized:
            normalized = normalized.replace(old_tone, new_tone)

    # Bước 4: Thay dấu câu bằng khoảng trắng
    normalized = PUNCTUATION_REGEX.sub(" ", normalized)

    # Bước 5: Chuẩn hóa khoảng trắng
    tokens = normalized.split()
    return " ".join(tokens)


def analyze_technical_terms(
    ground_truth: str,
    hypothesis: str,
    terms_list: list[str]
) -> TechnicalTermResult:
    """
    Phân tích độ chính xác nhận dạng các thuật ngữ kỹ thuật trong tập Code-Switching.

    Phân loại từng thuật ngữ xuất hiện trong Ground Truth thành:
    - correct: Xuất hiện nguyên vẹn trong bản nhận dạng (Hypothesis).
    - substitution: Xuất hiện dạng biến dạng hoặc sai từ liên quan (prefix match / edit distance).
    - deletion: Hoàn toàn vắng mặt trong bản nhận dạng.
    """
    norm_gt = normalize_vietnamese_text(ground_truth)
    norm_hyp = normalize_vietnamese_text(hypothesis)

    details = []
    total_in_gt = 0
    correct_count = 0
    substitution_count = 0
    deletion_count = 0

    hyp_words = norm_hyp.split()

    for raw_term in terms_list:
        norm_term = normalize_vietnamese_text(raw_term)
        if not norm_term:
            continue

        # Đếm số lần term xuất hiện trong Ground Truth
        term_words = norm_term.split()
        term_pattern = r"\b" + r"\s+".join(map(re.escape, term_words)) + r"\b"
        matches = len(re.findall(term_pattern, norm_gt))

        if matches > 0:
            total_in_gt += matches
            # Kiểm tra term trong Hypothesis
            hyp_matches = len(re.findall(term_pattern, norm_hyp))

            for _ in range(matches):
                if hyp_matches > 0:
                    correct_count += 1
                    hyp_matches -= 1
                    details.append({"term": raw_term, "status": "correct"})
                else:
                    # Kiểm tra xem có từ nào tương tự hoặc biến dạng trong hypothesis không
                    # (khớp tiền tố >= 3 ký tự hoặc một từ trong cụm xuất hiện)
                    is_sub = False
                    for tw in term_words:
                        for hw in hyp_words:
                            if tw == hw or (len(tw) >= 4 and hw.startswith(tw[:3])):
                                is_sub = True
                                break
                        if is_sub:
                            break

                    if is_sub:
                        substitution_count += 1
                        details.append({"term": raw_term, "status": "substitution"})
                    else:
                        deletion_count += 1
                        details.append({"term": raw_term, "status": "deletion"})

    # Tính toán Recall và Accuracy
    term_recall = (correct_count / total_in_gt) if total_in_gt > 0 else 1.0
    total_classified = correct_count + substitution_count + deletion_count
    term_accuracy = (correct_count / total_classified) if total_classified > 0 else 1.0

    return {
        "total_in_gt": total_in_gt,
        "correct": correct_count,
        "substitution": substitution_count,
        "deletion": deletion_count,
        "term_recall": round(term_recall, 4),
        "term_accuracy": round(term_accuracy, 4),
        "details": details
    }
