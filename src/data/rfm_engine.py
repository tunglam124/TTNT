"""Làm sạch dữ liệu và tính đặc trưng RFM + hành vi cho từng khách hàng."""
from __future__ import annotations

import pandas as pd


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Loại dòng thiếu khóa, số lượng/đơn giá <= 0, trùng lặp; tính LINE_TOTAL > 0.
    Không cắt ngoại lai ở mức dòng (xử lý ở mức khách hàng trong build_features)."""
    df = df.dropna(subset=["CUSTOMER_ID", "TRANSACTION_ID", "DATE"]).copy()
    df["CUSTOMER_ID"] = df["CUSTOMER_ID"].astype(str)
    df["TRANSACTION_ID"] = df["TRANSACTION_ID"].astype(str)
    for num_col in ["QUANTITY", "UNIT_PRICE", "AMOUNT", "PROFIT", "DISCOUNT"]:
        if num_col in df.columns and not pd.api.types.is_numeric_dtype(df[num_col]):
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
    if "QUANTITY" in df.columns:
        df = df[df["QUANTITY"] > 0]
    if "UNIT_PRICE" in df.columns:
        df = df[df["UNIT_PRICE"] > 0]
    df = df.drop_duplicates()
    if "LINE_TOTAL" not in df.columns:
        if "AMOUNT" in df.columns and df["AMOUNT"].notna().any():
            df["LINE_TOTAL"] = df["AMOUNT"]
        elif "QUANTITY" in df.columns and "UNIT_PRICE" in df.columns:
            df["LINE_TOTAL"] = df["QUANTITY"] * df["UNIT_PRICE"]
        else:
            raise ValueError("Cần cột AMOUNT hoặc cả QUANTITY và UNIT_PRICE để tính doanh thu.")
    return df[df["LINE_TOTAL"] > 0].reset_index(drop=True)


def use_days_for_frequency(df: pd.DataFrame, mode: str = "auto") -> bool:
    if mode == "days":
        return True
    if mode == "transaction":
        return False
    shared = (df.groupby("TRANSACTION_ID")["CUSTOMER_ID"].nunique() > 1).mean() * 100
    return shared > 5


def build_features(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None,
                   frequency_mode: str = "auto", winsor_pct: float | None = 0.99,
                   use_days: bool | None = None) -> pd.DataFrame:
    """Tính đặc trưng cho mỗi khách hàng tính đến `snapshot_date` (chỉ dùng giao dịch TRƯỚC mốc này).

    Trả về: Recency, Frequency, Monetary (+ Tenure, AvgOrderValue, AvgDaysBetween, Lines).
    """
    if snapshot_date is None:
        snapshot_date = df["DATE"].max() + pd.Timedelta(days=1)
    d = df[df["DATE"] < snapshot_date]
    if use_days is None:
        use_days = use_days_for_frequency(d, frequency_mode)

    d = d.assign(_DAY=d["DATE"].dt.normalize())
    g = d.groupby("CUSTOMER_ID")
    first, last = g["DATE"].min(), g["DATE"].max()
    freq = g["_DAY"].nunique() if use_days else g["TRANSACTION_ID"].nunique()

    feat = pd.DataFrame({
        "Recency": (snapshot_date - last).dt.days,
        "Frequency": freq,
        "Monetary": g["LINE_TOTAL"].sum(),
        "Tenure": (snapshot_date - first).dt.days,
        "Lines": g.size(),
    })
    feat["AvgOrderValue"] = feat["Monetary"] / feat["Frequency"]
    span = (last - first).dt.days
    feat["AvgDaysBetween"] = (span / (feat["Frequency"] - 1).where(feat["Frequency"] > 1)).fillna(-1)

    item_col = "CATEGORY" if ("CATEGORY" in d.columns and d["CATEGORY"].notna().any()) else (
        "PRODUCT" if ("PRODUCT" in d.columns and d["PRODUCT"].notna().any()) else None
    )
    if item_col:
        feat["CategoryCount"] = g[item_col].nunique()
        cat_spend = d.groupby(["CUSTOMER_ID", item_col])["LINE_TOTAL"].sum().reset_index()
        top_cat = (cat_spend.sort_values("LINE_TOTAL", ascending=False)
                   .drop_duplicates("CUSTOMER_ID").set_index("CUSTOMER_ID")[item_col])
        feat["TopCategory"] = top_cat
    if "CHANNEL" in d.columns and d["CHANNEL"].notna().any():
        ch_cnt = d.groupby(["CUSTOMER_ID", "CHANNEL"]).size().reset_index(name="n")
        pref_ch = (ch_cnt.sort_values("n", ascending=False)
                   .drop_duplicates("CUSTOMER_ID").set_index("CUSTOMER_ID")["CHANNEL"])
        feat["PreferredChannel"] = pref_ch
    if "PROFIT" in d.columns and pd.api.types.is_numeric_dtype(d["PROFIT"]):
        feat["Profit"] = g["PROFIT"].sum()
    if "DISCOUNT" in d.columns and pd.api.types.is_numeric_dtype(d["DISCOUNT"]):
        feat["AvgDiscount"] = g["DISCOUNT"].mean()

    if winsor_pct:
        cap = feat["Monetary"].quantile(winsor_pct)
        feat["Monetary"] = feat["Monetary"].clip(upper=cap)
        feat["AvgOrderValue"] = feat["AvgOrderValue"].clip(upper=cap)

    feat.attrs.update({"snapshot_date": snapshot_date, "frequency_by_days": use_days})
    return feat.reset_index()
