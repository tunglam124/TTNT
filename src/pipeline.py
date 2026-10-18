"""Pipeline huấn luyện end-to-end: dữ liệu -> đặc trưng RFM+ -> phân cụm -> dịch chuyển -> gợi ý bán chéo -> dự báo -> lưu registry."""
from __future__ import annotations

import sys
from datetime import datetime
from typing import Callable

from config.settings import settings
from src.data.ingestion import load_transactions
from src.data.rfm_engine import build_features, clean_transactions, use_days_for_frequency
from src.models.evaluation import cluster_metrics
from src.models.predictor import CustomerPredictor
from src.models.recommender import CrossSellRecommender
from src.models.registry import ModelRegistry, RunArtifacts
from src.models.segmentation import (
    SegmentationModel,
    compare_algorithms,
    compute_segment_migration,
    search_k,
    segment_profile,
)

ProgressFn = Callable[[float, str], None]


def _default_step(pct: float, msg: str) -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(f"[{pct:>5.0%}] {msg}", flush=True)


def run_pipeline(source=None, column_map: dict | None = None, final_k: int | None = None,
                 compare: bool = True, train_predictor: bool = True,
                 progress: ProgressFn | None = None) -> RunArtifacts:
    step = progress or _default_step
    source = source if source is not None else settings.data_path
    rs = settings.random_state

    step(0.05, "Đọc và kiểm tra dữ liệu")
    ing = load_transactions(source, column_map or settings.column_map)
    df = clean_transactions(ing.df)
    use_days = use_days_for_frequency(df, settings.frequency_mode)

    step(0.20, "Tính đặc trưng RFM+ và hành vi đa kênh")
    feats = build_features(df, frequency_mode=settings.frequency_mode,
                           winsor_pct=settings.monetary_winsor_pct, use_days=use_days)
    if ing.label_columns:
        feats = feats.merge(df.groupby("CUSTOMER_ID")[ing.label_columns].first().reset_index(),
                            on="CUSTOMER_ID", how="left")

    step(0.30, "Tìm số cụm tối ưu (Elbow + Silhouette)")
    tmp = SegmentationModel(k=2, random_state=rs)
    X = tmp.scaler.fit_transform(tmp.transform_input(feats))
    k_table = search_k(X, range(settings.k_min, settings.k_max + 1), settings.silhouette_sample, rs)
    best_k = int(k_table.loc[k_table["silhouette"].idxmax(), "k"])
    k = final_k or settings.final_k or best_k

    step(0.45, f"Huấn luyện K-Means với k={k}")
    seg = SegmentationModel(k=k, random_state=rs)
    feats["Cluster"] = seg.fit(feats)
    feats["Segment"] = feats["Cluster"].map(seg.label_map)
    X = seg.scale(feats)
    seg_metrics = cluster_metrics(X, feats["Cluster"].to_numpy(), settings.silhouette_sample, rs)
    feats[["x_r", "x_f", "x_m"]] = X

    step(0.52, "Phân tích dịch chuyển phân khúc & luật mua kèm (Market Basket)")
    migration = compute_segment_migration(df, feats, seg, settings.churn_horizon_days, use_days)
    recommender = CrossSellRecommender().fit(df, feats)

    comparison = None
    if compare:
        step(0.60, "So sánh 6 thuật toán phân cụm")
        comparison = compare_algorithms(X, k, settings.max_sample_heavy, rs)

    predictor, pred_metrics, pred_error = None, None, None
    if train_predictor:
        step(0.75, "Huấn luyện mô hình dự báo churn & giá trị tương lai")
        try:
            predictor = CustomerPredictor(horizon_days=settings.churn_horizon_days, random_state=rs)
            pred_metrics = predictor.fit(df, use_days)
            feats = feats.join(predictor.predict(feats))
        except ValueError as e:
            predictor, pred_error = None, str(e)

    step(0.90, "Lưu mô hình vào registry")
    registry = ModelRegistry(settings.models_dir)
    profile = segment_profile(feats)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source) if isinstance(source, str) else "upload",
        "data": ing.report,
        "mapping": ing.mapping,
        "label_columns": ing.label_columns,
        "frequency_by_days": bool(use_days),
        "snapshot_date": str(feats.attrs.get("snapshot_date", df["DATE"].max()))[:10],
        "k": int(k), "best_k_silhouette": best_k,
        "k_search": k_table.to_dict(orient="records"),
        "segmentation_metrics": seg_metrics,
        "segment_thresholds": seg.thresholds,
        "segment_profile": profile.reset_index().to_dict(orient="records"),
        "segment_migration": migration,
        "market_basket": {
            "available": recommender.available,
            "item_col": recommender.item_col,
            "top_rules": recommender.rules[:20],
            "segment_top_items": recommender.segment_top_items,
            "global_top_items": recommender.global_top_items[:15],
        },
        "algorithm_comparison": (comparison.reset_index().to_dict(orient="records")
                                 if comparison is not None else None),
        "predictor_metrics": pred_metrics,
        "predictor_error": pred_error,
    }
    opt_cols = ["QUANTITY", "UNIT_PRICE", "CATEGORY", "PRODUCT", "CHANNEL", "PROFIT", "DISCOUNT"]
    keep = ["CUSTOMER_ID", "TRANSACTION_ID", "DATE", "LINE_TOTAL"] + \
           [c for c in opt_cols if c in df.columns]
    art = RunArtifacts(run_id=registry.new_run_id(), segmentation=seg, predictor=predictor,
                       customers=feats, transactions=df[keep], metadata=metadata,
                       recommender=recommender)
    registry.save(art)
    step(1.0, f"Hoàn tất: {art.run_id}")
    return art


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Huấn luyện pipeline Customer Intelligence")
    ap.add_argument("--data", default=None, help="Đường dẫn CSV giao dịch")
    ap.add_argument("--k", type=int, default=None, help="Ép số cụm")
    ap.add_argument("--no-compare", action="store_true", help="Bỏ qua so sánh thuật toán")
    a = ap.parse_args()
    run_pipeline(a.data, final_k=a.k, compare=not a.no_compare)
