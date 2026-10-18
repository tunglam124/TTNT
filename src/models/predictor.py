"""Predictive AI: dự báo khả năng rời bỏ (churn) và giá trị chi tiêu tương lai (CLV ngắn hạn).

Vì dữ liệu không có nhãn churn đáng tin, nhãn được TẠO TỪ CHÍNH LỊCH SỬ GIAO DỊCH theo cách chia thời gian:

    |---------- quá khứ (tính đặc trưng) ----------| cutoff |--- horizon ngày (quan sát) ---| end
    churn = 1  nếu khách KHÔNG mua lần nào trong cửa sổ quan sát
    value      = tổng chi tiêu của khách trong cửa sổ quan sát

Mô hình học trên mốc `cutoff`, sau đó chấm điểm cho trạng thái hiện tại (mốc `end`) để dự báo horizon ngày TỚI.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (accuracy_score, average_precision_score, mean_absolute_error,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

from src.data.rfm_engine import build_features

MIN_RELIABLE_AUC = 0.60  # dưới ngưỡng này mô hình churn bị coi là không đáng tin (0.5 = đoán ngẫu nhiên)
FEATURES = ["Recency", "Frequency", "Monetary", "Tenure", "Lines", "AvgOrderValue", "AvgDaysBetween"]


def risk_level(p: float) -> str:
    return "Cao" if p >= 0.7 else ("Trung bình" if p >= 0.4 else "Thấp")


@dataclass
class CustomerPredictor:
    horizon_days: int = 90
    random_state: int = 42
    churn_model: HistGradientBoostingClassifier | None = None
    value_model: HistGradientBoostingRegressor | None = None
    metrics: dict = field(default_factory=dict)

    def build_training_set(self, df: pd.DataFrame, use_days: bool) -> pd.DataFrame:
        end = df["DATE"].max() + pd.Timedelta(days=1)
        cutoff = end - pd.Timedelta(days=self.horizon_days)
        X = build_features(df, snapshot_date=cutoff, use_days=use_days, winsor_pct=0.99)
        future = df[(df["DATE"] >= cutoff) & (df["DATE"] < end)]
        spend = future.groupby("CUSTOMER_ID")["LINE_TOTAL"].sum()
        X["future_value"] = X["CUSTOMER_ID"].map(spend).fillna(0.0)
        X["churn"] = (X["future_value"] == 0).astype(int)
        X.attrs.update({"cutoff": cutoff, "end": end})
        return X

    def fit(self, df: pd.DataFrame, use_days: bool) -> dict:
        data = self.build_training_set(df, use_days)
        if len(data) < 200 or data["churn"].nunique() < 2:
            raise ValueError("Không đủ dữ liệu lịch sử để huấn luyện mô hình dự báo "
                             f"(cần khách hàng mua trước mốc cutoff và cả hai lớp churn/không churn).")
        X, y, v = data[FEATURES], data["churn"], data["future_value"]
        X_tr, X_te, y_tr, y_te, v_tr, v_te = train_test_split(
            X, y, v, test_size=0.2, random_state=self.random_state, stratify=y)

        self.churn_model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=self.random_state).fit(X_tr, y_tr)
        p = self.churn_model.predict_proba(X_te)[:, 1]

        self.value_model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=self.random_state).fit(X_tr, v_tr)
        v_pred = np.clip(self.value_model.predict(X_te), 0, None)

        self.metrics = {
            "cutoff": str(data.attrs["cutoff"].date()),
            "horizon_days": self.horizon_days,
            "train_customers": int(len(X_tr)),
            "test_customers": int(len(X_te)),
            "churn_rate": float(y.mean()),
            "churn_auc": float(roc_auc_score(y_te, p)),
            "churn_pr_auc": float(average_precision_score(y_te, p)),
            "churn_accuracy": float(accuracy_score(y_te, p >= 0.5)),
            "churn_baseline_accuracy": float(max(y_te.mean(), 1 - y_te.mean())),
            "value_mae": float(mean_absolute_error(v_te, v_pred)),
            "value_baseline_mae": float(mean_absolute_error(v_te, np.full(len(v_te), v_tr.mean()))),
        }
        # Cổng kiểm định: mô hình chỉ được coi là đáng tin khi vượt rõ mức ngẫu nhiên / mức cơ sở
        self.metrics["churn_reliable"] = self.metrics["churn_auc"] >= MIN_RELIABLE_AUC
        self.metrics["value_reliable"] = (self.metrics["value_mae"]
                                          <= 0.95 * self.metrics["value_baseline_mae"])
        return self.metrics

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X = features[FEATURES]
        p = self.churn_model.predict_proba(X)[:, 1]
        v = np.clip(self.value_model.predict(X), 0, None)
        return pd.DataFrame({
            "churn_prob": p,
            "churn_risk": [risk_level(x) for x in p],
            "pred_value_90d": v,
        }, index=features.index)
