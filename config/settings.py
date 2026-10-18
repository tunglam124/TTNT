"""Cấu hình tập trung cho toàn hệ thống.

Mọi giá trị có thể ghi đè bằng biến môi trường hoặc file `.env` ở thư mục gốc dự án.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Đọc file .env đơn giản (KEY=VALUE), không ghi đè biến đã có sẵn."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Settings:
    # Dữ liệu
    data_path: str = os.getenv("DATA_PATH", r"D:\archive (6)\supermarket_large_dataset.csv")
    artifacts_dir: Path = Path(os.getenv("ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts")))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))

    # Phân cụm
    k_min: int = 2
    k_max: int = 8
    final_k: int | None = int(os.environ["FINAL_K"]) if os.getenv("FINAL_K") else None
    max_sample_heavy: int = 3000
    silhouette_sample: int = 10000
    frequency_mode: str = os.getenv("FREQUENCY_MODE", "auto")  # auto | transaction | days
    monetary_winsor_pct: float = 0.99

    # Dự báo
    churn_horizon_days: int = int(os.getenv("CHURN_HORIZON_DAYS", "90"))

    # Generative AI
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # API
    api_url: str = os.getenv("API_URL", "http://localhost:8000")

    column_map: dict = field(default_factory=lambda: {
        "customer_id": None, "transaction_id": None, "date": None,
        "quantity": None, "unit_price": None, "amount": None,
    })

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.gemini_api_key)


settings = Settings()
