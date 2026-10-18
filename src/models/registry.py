"""Model registry: lưu/nạp mô hình + dữ liệu kết quả theo từng lần chạy (run), có con trỏ `latest`."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd


@dataclass
class RunArtifacts:
    run_id: str
    segmentation: object
    predictor: object | None
    customers: pd.DataFrame
    transactions: pd.DataFrame
    metadata: dict
    recommender: object | None = None


class ModelRegistry:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def new_run_id(self) -> str:
        return datetime.now().strftime("run_%Y%m%d_%H%M%S")

    def save(self, art: RunArtifacts) -> Path:
        d = self.models_dir / art.run_id
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"segmentation": art.segmentation, "predictor": art.predictor, "recommender": art.recommender},
            d / "models.joblib",
        )
        art.customers.to_pickle(d / "customers.pkl")
        art.transactions.to_pickle(d / "transactions.pkl")
        (d / "metadata.json").write_text(json.dumps(art.metadata, ensure_ascii=False, indent=2, default=str),
                                         encoding="utf-8")
        art.customers.to_csv(d / "customers_scored.csv", index=False)
        (self.models_dir / "LATEST").write_text(art.run_id, encoding="utf-8")
        return d

    def list_runs(self) -> list[dict]:
        runs = []
        for d in sorted(self.models_dir.glob("run_*"), reverse=True):
            meta_file = d / "metadata.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                runs.append({"run_id": d.name, "created_at": meta.get("created_at"),
                             "k": meta.get("k"), "customers": meta.get("data", {}).get("customers")})
        return runs

    def latest_id(self) -> str | None:
        p = self.models_dir / "LATEST"
        return p.read_text(encoding="utf-8").strip() if p.exists() else None

    def load(self, run_id: str | None = None) -> RunArtifacts | None:
        run_id = run_id or self.latest_id()
        if not run_id or not (self.models_dir / run_id).exists():
            return None
        d = self.models_dir / run_id
        models = joblib.load(d / "models.joblib")
        return RunArtifacts(
            run_id=run_id,
            segmentation=models["segmentation"],
            predictor=models["predictor"],
            customers=pd.read_pickle(d / "customers.pkl"),
            transactions=pd.read_pickle(d / "transactions.pkl"),
            metadata=json.loads((d / "metadata.json").read_text(encoding="utf-8")),
            recommender=models.get("recommender"),
        )

    def set_latest(self, run_id: str) -> None:
        if not (self.models_dir / run_id).exists():
            raise ValueError(f"Không tồn tại run {run_id}")
        (self.models_dir / "LATEST").write_text(run_id, encoding="utf-8")
