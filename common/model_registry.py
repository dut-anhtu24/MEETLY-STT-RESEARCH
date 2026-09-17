"""
Module đăng ký và quản lý metadata các mô hình STT (Model Registry Skeleton).

Cung cấp:
- Lớp dữ liệu ModelMetadata chứa thông tin kiến trúc, backends hỗ trợ, và cờ trạng thái kiểm thử.
- Khung nạp cấu hình models.yaml và lọc ứng viên (loại trừ model smoke-test khỏi benchmark chính).
- Giao diện trừu tượng BaseSTTModel cho các engine suy luận sau này.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class ModelMetadata:
    """Metadata mô tả đặc tính kỹ thuật của một mô hình STT."""
    name: str
    model_id: str
    family: str
    architecture: str
    target_languages: list[str] = field(default_factory=list)
    supported_backends: list[str] = field(default_factory=list)
    preferred_backend: str = "transformers"
    conversion_required: bool = False
    compatibility_status: str = "verified"  # verified, pending_verification, unsupported
    is_smoke_test: bool = False
    approx_params_m: float = 0.0
    default_precision: str = "float16"


class BaseSTTModel(ABC):
    """Lớp cơ sở trừu tượng cho tất cả các engine suy luận STT (Transformers, faster-whisper, CTC)."""

    def __init__(self, metadata: ModelMetadata, device: str = "cpu", precision: str = "float32"):
        self.metadata = metadata
        self.device = device
        self.precision = precision

    @abstractmethod
    def load(self) -> None:
        """Nạp trọng số mô hình vào bộ nhớ RAM / VRAM."""
        pass

    @abstractmethod
    def transcribe(self, audio: Any) -> str:
        """Thực hiện nhận dạng giọng nói trên mảng audio 16kHz."""
        pass


class ModelRegistry:
    """Bộ quản lý danh mục mô hình dựa trên file cấu hình models.yaml."""

    def __init__(self, config_path: str | Path | None = None):
        self.catalog: dict[str, ModelMetadata] = {}
        if config_path is not None:
            self.load_from_yaml(config_path)

    def load_from_yaml(self, config_path: str | Path) -> None:
        """Đọc và nạp danh mục model từ file YAML."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file cấu hình model: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        models_dict = data.get("models", {})
        self.catalog.clear()

        for key, info in models_dict.items():
            meta = ModelMetadata(
                name=info.get("name", key),
                model_id=info.get("model_id", key),
                family=info.get("family", "Unknown"),
                architecture=info.get("architecture", "encoder_decoder"),
                target_languages=info.get("target_languages", ["vi"]),
                supported_backends=info.get("supported_backends", ["transformers"]),
                preferred_backend=info.get("preferred_backend", "transformers"),
                conversion_required=info.get("conversion_required", False),
                compatibility_status=info.get("compatibility_status", "verified"),
                is_smoke_test=info.get("is_smoke_test", False),
                approx_params_m=float(info.get("approx_params_m", 0.0)),
                default_precision=info.get("default_precision", "float16")
            )
            self.catalog[key] = meta

    def get(self, model_key: str) -> ModelMetadata:
        """Lấy metadata của một model theo key. Báo lỗi nếu không tồn tại."""
        if model_key not in self.catalog:
            raise KeyError(f"Mô hình '{model_key}' không có trong registry. Có sẵn: {list(self.catalog.keys())}")
        return self.catalog[model_key]

    def list_candidates(self, include_smoke_test: bool = False) -> list[ModelMetadata]:
        """
        Liệt kê danh sách các mô hình ứng viên.

        :param include_smoke_test: Nếu False, tự động loại bỏ các model phục vụ smoke-test (whisper-tiny).
        """
        return [
            meta for meta in self.catalog.values()
            if include_smoke_test or not meta.is_smoke_test
        ]

    def is_smoke_test_model(self, model_key: str) -> bool:
        """Kiểm tra xem model có phải là model thử nghiệm nội bộ hay không."""
        return self.get(model_key).is_smoke_test
