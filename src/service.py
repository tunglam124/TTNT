"""Tầng dịch vụ dùng chung cho FastAPI và Streamlit: đọc run mới nhất từ registry và trả dữ liệu nghiệp vụ."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config.settings import settings
from src.ai_agent.copilot import PLAYBOOK, MarketingCopilot
from src.models.registry import ModelRegistry, RunArtifacts
from src.models.segmentation import RFM_COLS, base_segment

_registry = ModelRegistry(settings.models_dir)
_cache: dict = {}


class NoModelError(RuntimeError):
    pass


def registry() -> ModelRegistry:
    return _registry


def get_run(run_id: str | None = None) -> RunArtifacts:
    rid = run_id or _registry.latest_id()
    if rid is None:
        raise NoModelError("Chưa có mô hình nào. Hãy chạy huấn luyện trước (trang Quản lý hoặc POST /api/train).")
    if rid not in _cache:
        art = _registry.load(rid)
        if art is None:
            raise NoModelError(f"Không tìm thấy run {rid}")
        _cache.clear()
        _cache[rid] = art
    return _cache[rid]


def invalidate_cache() -> None:
    _cache.clear()


def _clean(obj):
    """Chuyển NaN/numpy về kiểu JSON hợp lệ."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return obj


def prediction_status(art: RunArtifacts | None = None) -> dict:
    art = art or get_run()
    m = art.metadata.get("predictor_metrics") or {}
    return {"available": art.predictor is not None,
            "churn_reliable": bool(m.get("churn_reliable")),
            "value_reliable": bool(m.get("value_reliable")),
            "metrics": m, "error": art.metadata.get("predictor_error")}


def overview() -> dict:
    art = get_run()
    c = art.customers
    md = art.metadata
    return _clean({
        "run_id": art.run_id,
        "created_at": md["created_at"],
        "snapshot_date": md["snapshot_date"],
        "customers": len(c),
        "revenue": float(c["Monetary"].sum()),
        "profit": float(c["Profit"].sum()) if "Profit" in c.columns else None,
        "avg_monetary": float(c["Monetary"].mean()),
        "avg_frequency": float(c["Frequency"].mean()),
        "active_90d_pct": float((c["Recency"] <= 90).mean() * 100),
        "k": md["k"],
        "segmentation_metrics": md["segmentation_metrics"],
        "data": md["data"],
        "prediction": prediction_status(art),
    })


def totals() -> dict:
    art = get_run()
    return {"customers": int(len(art.customers)), "revenue": float(art.customers["Monetary"].sum()),
            "horizon": settings.churn_horizon_days}


def segment_profiles(for_llm: bool = False) -> list[dict]:
    """Hồ sơ phân khúc. Bỏ số liệu dự báo nếu mô hình không đạt cổng kiểm định (tránh đưa nhiễu cho LLM)."""
    art = get_run()
    profiles = [dict(p) for p in art.metadata["segment_profile"]]
    st = prediction_status(art)
    mb = (art.metadata.get("market_basket") or {}).get("segment_top_items") or {}
    for p in profiles:
        if not st["churn_reliable"]:
            p.pop("churn_prob_mean", None)
        if not st["value_reliable"]:
            p.pop("pred_value_90d_mean", None)
        seg_items = mb.get(p["Segment"], [])
        p["top_categories"] = [x["item"] for x in seg_items[:3]]
    return _clean(profiles)


def segment_migration() -> dict:
    art = get_run()
    return _clean(art.metadata.get("segment_migration") or {"available": False})


def market_basket() -> dict:
    art = get_run()
    mb = dict(art.metadata.get("market_basket") or {"available": False})
    if "rules" not in mb:
        mb["rules"] = mb.get("top_rules") or (
            art.recommender.rules if getattr(art, "recommender", None) is not None else []
        )
    return _clean(mb)


def list_customers(segment: str | None = None, limit: int = 100, offset: int = 0,
                   sort_by: str = "Monetary", ascending: bool = False) -> dict:
    c = get_run().customers
    if segment:
        c = c[c["Segment"] == segment]
    if sort_by in c.columns:
        c = c.sort_values(sort_by, ascending=ascending)
    extra = [x for x in ["TopCategory", "CategoryCount", "PreferredChannel", "Profit",
                         "churn_prob", "churn_risk", "pred_value_90d"] if x in c.columns]
    cols = ["CUSTOMER_ID", "Segment", *RFM_COLS, *extra]
    page = c[cols].iloc[offset: offset + limit]
    return _clean({"total": int(len(c)), "items": page.to_dict(orient="records")})


def explain_customer_segment(row: pd.Series, thresholds: dict) -> list[str]:
    """Giải thích vì sao khách hàng được xếp vào phân khúc hiện tại (Explainable AI)."""
    reasons = []
    r_cut = thresholds.get("recency_p33", 90)
    f_cut = thresholds.get("frequency_mean", 1.5)
    m_cut = thresholds.get("monetary_mean", 100.0)

    if row["Recency"] <= r_cut:
        reasons.append(f"✅ Mua gần đây ({int(row['Recency'])} ngày ≤ ngưỡng tích cực {r_cut:.0f} ngày)")
    else:
        reasons.append(f"⚠️ Đã lâu chưa mua ({int(row['Recency'])} ngày > ngưỡng tích cực {r_cut:.0f} ngày)")

    if row["Frequency"] >= f_cut:
        reasons.append(f"✅ Tần suất mua cao ({int(row['Frequency'])} lần ≥ TB hệ thống {f_cut:.1f})")
    else:
        reasons.append(f"⚠️ Tần suất mua thấp ({int(row['Frequency'])} lần < TB hệ thống {f_cut:.1f})")

    if row["Monetary"] >= m_cut:
        reasons.append(f"✅ Chi tiêu cao ({row['Monetary']:,.1f} ≥ TB hệ thống {m_cut:,.1f})")
    else:
        reasons.append(f"⚠️ Chi tiêu dưới mức TB ({row['Monetary']:,.1f} < TB hệ thống {m_cut:,.1f})")
    return reasons


def customer_detail(customer_id: str) -> dict:
    art = get_run()
    c = art.customers
    row = c[c["CUSTOMER_ID"] == str(customer_id)]
    if row.empty:
        raise KeyError(customer_id)
    r = row.iloc[0]

    # 5 chiều bách phân vị (0-100) cho biểu đồ Radar cá nhân vs trung bình cụm
    pct = {col: float((c[col] <= r[col]).mean() * 100) for col in RFM_COLS}
    pct["Recency"] = 100 - pct["Recency"]
    aov_col = c["AvgOrderValue"] if "AvgOrderValue" in c.columns else c["Monetary"] / c["Frequency"].clip(lower=1)
    r_aov = float(r.get("AvgOrderValue", r["Monetary"] / max(r["Frequency"], 1)))
    pct["AvgOrderValue"] = float((aov_col <= r_aov).mean() * 100)
    if "CategoryCount" in c.columns:
        pct["Diversity"] = float((c["CategoryCount"] <= r["CategoryCount"]).mean() * 100)
    elif "Tenure" in c.columns:
        pct["Diversity"] = float((c["Tenure"] <= r["Tenure"]).mean() * 100)
    else:
        pct["Diversity"] = pct["Frequency"]
    pct["CategoryCount"] = pct["Diversity"]

    tx = art.transactions[art.transactions["CUSTOMER_ID"] == str(customer_id)].sort_values("DATE")

    # Gợi ý ngành hàng bán chéo (Next-Best-Category)
    recs = []
    bought_cats = []
    if "CATEGORY" in tx.columns and tx["CATEGORY"].notna().any():
        bought_cats = (tx.groupby("CATEGORY")["LINE_TOTAL"].sum()
                       .sort_values(ascending=False).index.astype(str).tolist())
    if getattr(art, "recommender", None) is not None and art.recommender.available:
        recs = art.recommender.recommend_for_customer(bought_cats, segment=str(r["Segment"]), top_n=3)
        for item in recs:
            if "recommended_category" not in item and "category" in item:
                item["recommended_category"] = item["category"]

    thresholds = art.metadata.get("segment_thresholds") or {}
    explanation = explain_customer_segment(r, thresholds)

    profiles = art.metadata.get("segment_profile") or []
    seg_bench = next((p for p in profiles if p.get("Segment") == str(r["Segment"])), {})

    hide = {"x_r", "x_f", "x_m"}
    st = prediction_status(art)
    data = {k: v for k, v in r.to_dict().items() if k not in hide}
    return _clean({
        "customer": data,
        "percentiles": pct,
        "segment_benchmark": seg_bench,
        "purchased_categories": bought_cats,
        "recommendations": recs,
        "explanation": explanation,
        "transactions": tx.assign(DATE=tx["DATE"].astype(str)).to_dict(orient="records"),
        "prediction_reliable": st["churn_reliable"],
    })


def score_rfm(records: list[dict]) -> list[dict]:
    """Chấm phân khúc (và dự báo nếu có) cho khách hàng mới từ đặc trưng RFM nhập vào."""
    art = get_run()
    df = pd.DataFrame(records)
    for col in ["Tenure", "Lines", "AvgOrderValue", "AvgDaysBetween"]:
        if col not in df.columns:
            df[col] = {"Tenure": df["Recency"], "Lines": df["Frequency"],
                       "AvgOrderValue": df["Monetary"] / df["Frequency"].clip(lower=1),
                       "AvgDaysBetween": -1}[col]
    out = df.join(art.segmentation.predict(df))
    if art.predictor is not None and prediction_status(art)["churn_reliable"]:
        out = out.join(art.predictor.predict(df))
    return _clean(out.to_dict(orient="records"))


def export_campaign_targets(segment_name: str, channel: str = "Email", limit: int = 5000) -> pd.DataFrame:
    """Tạo bảng danh sách khách hàng mục tiêu kèm mã voucher, ngành hàng gợi ý và tin nhắn trộn sẵn."""
    art = get_run()
    c = art.customers[art.customers["Segment"] == segment_name].copy()
    c = c.sort_values("Monetary", ascending=False).head(limit)

    play = PLAYBOOK.get(base_segment(segment_name), PLAYBOOK["Potential Customers"])
    disc_pct = int(play["discount"] * 100)
    prefix = "".join(w[0] for w in base_segment(segment_name).split()[:2]).upper()

    mb = (art.metadata.get("market_basket") or {}).get("segment_top_items") or {}
    seg_cats = [x["item"] for x in mb.get(segment_name, [])] or ["Sản phẩm nổi bật"]

    def _pick_rec_cat(top_cat):
        for cand in seg_cats:
            if str(cand) != str(top_cat):
                return cand
        return seg_cats[0]

    top_cat_series = c["TopCategory"].fillna(seg_cats[0]) if "TopCategory" in c.columns else pd.Series(seg_cats[0], index=c.index)
    rec_cat_series = top_cat_series.map(_pick_rec_cat)
    voucher_codes = [f"{prefix}{disc_pct}-{str(cid)[-4:]}" for cid in c["CUSTOMER_ID"]]

    messages = [
        f"Chào {cid}, tặng bạn mã {vcode} (giảm {disc_pct}%) khi mua ngành hàng {rcat} qua {channel}!"
        for cid, vcode, rcat in zip(c["CUSTOMER_ID"], voucher_codes, rec_cat_series)
    ]

    out = pd.DataFrame({
        "CUSTOMER_ID": c["CUSTOMER_ID"].to_numpy(),
        "Segment": c["Segment"].to_numpy(),
        "Recency_Days": c["Recency"].to_numpy(),
        "Frequency": c["Frequency"].to_numpy(),
        "Monetary": c["Monetary"].round(2).to_numpy(),
        "Favorite_Category": top_cat_series.to_numpy(),
        "Recommended_CrossSell": rec_cat_series.to_numpy(),
        "Contact_Channel": (c["PreferredChannel"].fillna(channel).to_numpy()
                            if "PreferredChannel" in c.columns else channel),
        "Voucher_Code": voucher_codes,
        "Discount_Pct": f"{disc_pct}%",
        "Personalized_Message": messages,
    })
    return out


def copilot() -> MarketingCopilot:
    return MarketingCopilot()
