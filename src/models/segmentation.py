"""Phân khúc khách hàng: chuẩn hóa RFM, tự chọn k, K-Means, đặt tên cụm theo hồ sơ, so sánh thuật toán."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, MeanShift, SpectralClustering, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .evaluation import cluster_metrics, dunn_index, fuzzy_c_means

RFM_COLS = ["Recency", "Frequency", "Monetary"]

SEGMENT_INFO = {
    "Best Customers": "Mua gần đây, thường xuyên và chi tiêu cao — nhóm giá trị nhất.",
    "Loyal Customers": "Mua gần đây và có tần suất hoặc chi tiêu tốt.",
    "Potential Customers": "Mới mua gần đây nhưng tần suất/chi tiêu còn thấp — cần nuôi dưỡng.",
    "At-Risk Customers": "Từng mua nhiều/chi tiêu tốt nhưng đã lâu không quay lại.",
    "Lost Customers": "Lâu không mua, tần suất và chi tiêu thấp.",
}


def base_segment(name: str) -> str:
    return name.split(" (")[0]


@dataclass
class SegmentationModel:
    k: int
    random_state: int = 42
    scaler: StandardScaler = field(default_factory=StandardScaler)
    kmeans: KMeans | None = None
    label_map: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)

    @staticmethod
    def transform_input(rfm: pd.DataFrame) -> np.ndarray:
        return np.log1p(rfm[RFM_COLS].clip(lower=0).to_numpy(dtype=float))

    def fit(self, rfm: pd.DataFrame) -> np.ndarray:
        X = self.scaler.fit_transform(self.transform_input(rfm))
        self.kmeans = KMeans(n_clusters=self.k, random_state=self.random_state, n_init=10).fit(X)
        labels = self.kmeans.labels_
        self._name_clusters(rfm, labels)
        return labels

    def scale(self, rfm: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(self.transform_input(rfm))

    def predict(self, rfm: pd.DataFrame) -> pd.DataFrame:
        labels = self.kmeans.predict(self.scale(rfm))
        return pd.DataFrame({"Cluster": labels, "Segment": [self.label_map[l] for l in labels]},
                            index=rfm.index)

    def _name_clusters(self, rfm: pd.DataFrame, labels: np.ndarray) -> None:
        """Đặt tên theo hồ sơ R/F/M của cụm so với toàn bộ khách (không phụ thuộc k)."""
        self.thresholds = {"recency_p33": float(rfm["Recency"].quantile(0.33)),
                           "frequency_mean": float(rfm["Frequency"].mean()),
                           "monetary_mean": float(rfm["Monetary"].mean())}
        prof = rfm.assign(Cluster=labels).groupby("Cluster")[RFM_COLS].mean()
        names = {}
        for cid, row in prof.iterrows():
            recent = row["Recency"] <= self.thresholds["recency_p33"]
            freq = row["Frequency"] >= self.thresholds["frequency_mean"]
            money = row["Monetary"] >= self.thresholds["monetary_mean"]
            if recent and freq and money:
                names[cid] = "Best Customers"
            elif recent and (freq or money):
                names[cid] = "Loyal Customers"
            elif recent:
                names[cid] = "Potential Customers"
            elif freq or money:
                names[cid] = "At-Risk Customers"
            else:
                names[cid] = "Lost Customers"
        score = prof["Monetary"].rank() + prof["Frequency"].rank() - prof["Recency"].rank()
        seen: dict = {}
        for cid in score.sort_values(ascending=False).index:
            seen[names[cid]] = seen.get(names[cid], 0) + 1
            if seen[names[cid]] > 1:
                names[cid] = f"{names[cid]} ({seen[names[cid]]})"
        self.label_map = {int(k): v for k, v in names.items()}


def search_k(X: np.ndarray, k_range=range(2, 9), sil_sample: int = 10000,
             random_state: int = 42) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
        m = cluster_metrics(X, km.labels_, sil_sample, random_state)
        rows.append({"k": k, "wcss": float(km.inertia_), **m})
    return pd.DataFrame(rows)


def segment_profile(rfm: pd.DataFrame) -> pd.DataFrame:
    """Bảng tổng hợp theo phân khúc (cần cột Segment) kèm điểm Radar chuẩn hóa (0-100)."""
    grp = rfm.groupby("Segment")
    prof = grp.agg(
        customers=("CUSTOMER_ID", "count"),
        recency_mean=("Recency", "mean"),
        frequency_mean=("Frequency", "mean"),
        monetary_mean=("Monetary", "mean"),
        monetary_total=("Monetary", "sum"),
    )
    prof["customer_pct"] = prof["customers"] / prof["customers"].sum() * 100
    prof["revenue_pct"] = prof["monetary_total"] / prof["monetary_total"].sum() * 100
    if "AvgOrderValue" in rfm.columns:
        prof["aov_mean"] = grp["AvgOrderValue"].mean()
    else:
        prof["aov_mean"] = prof["monetary_mean"] / prof["frequency_mean"].clip(lower=1)
    if "Profit" in rfm.columns:
        prof["profit_mean"] = grp["Profit"].mean()
        prof["profit_total"] = grp["Profit"].sum()
        tot_p = max(float(prof["profit_total"].sum()), 1.0)
        prof["profit_pct"] = prof["profit_total"] / tot_p * 100
    if "CategoryCount" in rfm.columns:
        prof["category_count_mean"] = grp["CategoryCount"].mean()
    if "TopCategory" in rfm.columns:
        prof["top_category"] = grp["TopCategory"].agg(
            lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "—"
        )
    if "PreferredChannel" in rfm.columns:
        prof["preferred_channel"] = grp["PreferredChannel"].agg(
            lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "—"
        )
    for col in ["churn_prob", "pred_value_90d"]:
        if col in rfm.columns:
            prof[f"{col}_mean"] = grp[col].mean()

    # Điểm Radar (trung bình bách phân vị 0-100 của khách trong từng cụm)
    tmp = pd.DataFrame({"Segment": rfm["Segment"]})
    tmp["radar_recency"] = (1.0 - rfm["Recency"].rank(pct=True)) * 100
    tmp["radar_frequency"] = rfm["Frequency"].rank(pct=True) * 100
    tmp["radar_monetary"] = rfm["Monetary"].rank(pct=True) * 100
    aov_s = rfm["AvgOrderValue"] if "AvgOrderValue" in rfm.columns else rfm["Monetary"] / rfm["Frequency"].clip(lower=1)
    tmp["radar_aov"] = aov_s.rank(pct=True) * 100
    div_s = rfm["CategoryCount"] if "CategoryCount" in rfm.columns else (
        rfm["Tenure"] if "Tenure" in rfm.columns else rfm["Frequency"]
    )
    tmp["radar_diversity"] = div_s.rank(pct=True) * 100
    radar_grp = tmp.groupby("Segment").mean()
    for c in ["radar_recency", "radar_frequency", "radar_monetary", "radar_aov", "radar_diversity"]:
        prof[c] = radar_grp[c]

    return prof.sort_values("monetary_mean", ascending=False).round(3)


def compute_segment_migration(df: pd.DataFrame, current_customers: pd.DataFrame,
                              seg_model: SegmentationModel, horizon_days: int = 90,
                              use_days: bool = False) -> dict:
    """Phân tích dịch chuyển phân khúc giữa mốc `cutoff = snapshot - horizon_days` và hiện tại."""
    from src.data.rfm_engine import build_features

    end = df["DATE"].max() + pd.Timedelta(days=1)
    cutoff = end - pd.Timedelta(days=horizon_days)
    past_df = df[df["DATE"] < cutoff]
    if past_df.empty:
        return {"available": False}

    past_feats = build_features(df, snapshot_date=cutoff, use_days=use_days, winsor_pct=0.99)
    past_preds = seg_model.predict(past_feats)
    past_map = pd.Series(past_preds["Segment"].to_numpy(), index=past_feats["CUSTOMER_ID"].to_numpy())

    cur = current_customers[["CUSTOMER_ID", "Segment", "Monetary"]].copy()
    cur["Past_Segment"] = cur["CUSTOMER_ID"].map(past_map).fillna("Khách mới (trong kỳ)")

    flow_df = cur.groupby(["Past_Segment", "Segment"]).agg(
        customers=("CUSTOMER_ID", "count"),
        revenue=("Monetary", "sum"),
    ).reset_index()
    src_totals = flow_df.groupby("Past_Segment")["customers"].transform("sum").clip(lower=1)
    flow_df["pct_of_source"] = (flow_df["customers"] / src_totals * 100).round(1)

    flows = [
        {"source": str(r["Past_Segment"]), "target": str(r["Segment"]),
         "customers": int(r["customers"]), "pct_of_source": float(r["pct_of_source"]),
         "revenue": round(float(r["revenue"]), 2)}
        for _, r in flow_df.iterrows()
    ]

    # Tóm tắt kinh doanh
    new_cnt = int((cur["Past_Segment"] == "Khách mới (trong kỳ)").sum())
    existing = cur[cur["Past_Segment"] != "Khách mới (trong kỳ)"]
    retained_same = int((existing["Past_Segment"] == existing["Segment"]).sum())
    slipped_to_risk = int(existing[
        existing["Past_Segment"].str.contains("Best|Loyal|Potential", regex=True)
        & existing["Segment"].str.contains("At-Risk|Lost", regex=True)
    ].shape[0])
    won_back = int(existing[
        existing["Past_Segment"].str.contains("At-Risk|Lost", regex=True)
        & existing["Segment"].str.contains("Best|Loyal|Potential", regex=True)
    ].shape[0])

    return {
        "available": True,
        "cutoff_date": str(cutoff.date()),
        "snapshot_date": str(end.date()),
        "horizon_days": int(horizon_days),
        "flows": flows,
        "summary": {
            "total_customers": int(len(cur)),
            "new_customers": new_cnt,
            "existing_customers": int(len(existing)),
            "retained_same_segment": retained_same,
            "slipped_to_risk_or_lost": slipped_to_risk,
            "reactivated_won_back": won_back,
        },
    }


def compare_algorithms(X: np.ndarray, k: int, max_sample: int = 3000,
                       random_state: int = 42) -> pd.DataFrame:
    """So sánh 6 thuật toán; mọi chỉ số chấm trên CÙNG một mẫu để công bằng."""
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), min(max_sample, len(X)), replace=False)
    Xs = X[idx]
    algos: dict = {
        "K-Means": lambda: KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X)[idx],
        "GMM": lambda: GaussianMixture(n_components=k, random_state=random_state).fit_predict(X)[idx],
        "Fuzzy C-Means": lambda: fuzzy_c_means(X, k, random_state=random_state)[idx],
        "DBSCAN": lambda: DBSCAN(eps=0.5, min_samples=10).fit_predict(X)[idx],
        "Spectral Clustering": lambda: SpectralClustering(
            n_clusters=k, affinity="nearest_neighbors", n_neighbors=10,
            assign_labels="kmeans", random_state=random_state).fit_predict(Xs),
        "Mean Shift": lambda: MeanShift(
            bandwidth=estimate_bandwidth(Xs, quantile=0.2, random_state=random_state) or None
        ).fit_predict(Xs),
    }
    rows = []
    for name, fn in algos.items():
        try:
            lb = fn()
        except Exception as e:  # thuật toán lỗi không làm hỏng cả bảng
            rows.append({"algorithm": name, "error": str(e)})
            continue
        m = cluster_metrics(Xs, lb)
        rows.append({"algorithm": name, "n_clusters": len(set(lb) - {-1}),
                     "noise_pct": float((lb == -1).mean() * 100), **m,
                     "dunn": dunn_index(Xs, lb)})
    res = pd.DataFrame(rows).set_index("algorithm")
    valid = res.dropna(subset=["silhouette"])
    if len(valid):
        rank = (valid["silhouette"].rank(ascending=False) + valid["davies_bouldin"].rank()
                + valid["calinski_harabasz"].rank(ascending=False))
        res["recommended"] = res.index == rank.idxmin()
    return res
