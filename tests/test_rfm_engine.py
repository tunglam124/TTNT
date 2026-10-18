"""Kiểm thử tính toán chỉ số RFM và đặc trưng hành vi."""
import pandas as pd
import pytest

from src.data.rfm_engine import build_features, clean_transactions


@pytest.fixture
def sample_clean_df():
    df = pd.DataFrame({
        "CUSTOMER_ID": ["C1", "C1", "C2", "C3"],
        "TRANSACTION_ID": ["T1", "T2", "T3", "T4"],
        "DATE": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01", "2024-03-01"]),
        "LINE_TOTAL": [100.0, 200.0, 50.0, 300.0],
    })
    return df


def test_clean_transactions():
    raw = pd.DataFrame({
        "CUSTOMER_ID": ["C1", None, "C2"],
        "TRANSACTION_ID": ["T1", "T2", "T3"],
        "DATE": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "LINE_TOTAL": [100.0, 50.0, -10.0],  # -10 phải bị loại
    })
    cleaned = clean_transactions(raw)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["CUSTOMER_ID"] == "C1"


def test_build_features(sample_clean_df):
    snapshot = pd.Timestamp("2024-03-02")
    feats = build_features(sample_clean_df, snapshot_date=snapshot, use_days=False)
    assert len(feats) == 3

    c1 = feats[feats["CUSTOMER_ID"] == "C1"].iloc[0]
    assert c1["Frequency"] == 2
    assert c1["Monetary"] == 300.0
    assert c1["Recency"] == (snapshot - pd.Timestamp("2024-01-15")).days


def test_transaction_id_priority_and_optional_columns():
    from src.data.ingestion import resolve_columns

    cols = ["customer_id", "basket_id", "transaction_id", "transaction_date", "total_amount", "product_category", "sales_channel"]
    mapping = resolve_columns(cols)
    # transaction_id phải được ưu tiên hơn basket_id
    assert mapping["transaction_id"] == "transaction_id"
    assert mapping["category"] == "product_category"
    assert mapping["channel"] == "sales_channel"


def test_rfm_plus_and_cross_sell_recommender():
    from src.models.recommender import CrossSellRecommender

    df = pd.DataFrame({
        "CUSTOMER_ID": ["C1", "C1", "C2", "C2", "C3", "C3"],
        "TRANSACTION_ID": ["T1", "T2", "T3", "T4", "T5", "T6"],
        "DATE": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-05", "2024-01-12", "2024-01-08", "2024-01-15"]),
        "LINE_TOTAL": [100.0, 150.0, 200.0, 120.0, 90.0, 110.0],
        "CATEGORY": ["Baby Food", "Diapers", "Baby Food", "Diapers", "Tea", "Grocery"],
        "PRODUCT": ["Milk A", "Diaper B", "Milk A", "Diaper B", "Green Tea", "Rice"],
        "CHANNEL": ["Mobile App", "Mobile App", "In Store", "In Store", "Mobile App", "In Store"],
        "PROFIT": [20.0, 30.0, 40.0, 25.0, 15.0, 22.0],
    })
    feats = build_features(df, snapshot_date=pd.Timestamp("2024-02-01"), use_days=False)
    assert "CategoryCount" in feats.columns
    assert "TopCategory" in feats.columns
    assert "PreferredChannel" in feats.columns
    assert "Profit" in feats.columns

    feats["Segment"] = ["Best Customers", "Best Customers", "Potential Customers"]
    rec = CrossSellRecommender().fit(df, feats)
    assert rec.available is True
    assert len(rec.rules) > 0
    recs_for_c1 = rec.recommend_for_customer(["Baby Food"], segment="Best Customers", top_n=2)
    assert len(recs_for_c1) >= 1
    assert recs_for_c1[0]["recommended_category"] == "Diapers"

