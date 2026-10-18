"""Kiểm thử tích hợp các endpoint của FastAPI."""
from fastapi.testclient import TestClient
import pytest

from api.main import app

client = TestClient(app)


def test_api_root():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "online"


def test_api_overview():
    res = client.get("/api/overview")
    assert res.status_code == 200
    data = res.json()
    assert "customers" in data
    assert "revenue" in data


def test_api_segments():
    res = client.get("/api/segments")
    assert res.status_code == 200
    segs = res.json()
    assert isinstance(segs, list)
    assert len(segs) >= 2


def test_api_score():
    res = client.post(
        "/api/score",
        json={"records": [{"Recency": 15.0, "Frequency": 3.0, "Monetary": 200.0}]},
    )
    assert res.status_code == 200
    scored = res.json()
    assert len(scored) == 1
    assert "Segment" in scored[0]


def test_api_campaign():
    res = client.post(
        "/api/ai/campaign",
        json={"segment_name": "Best Customers", "budget": 10000000},
    )
    assert res.status_code == 200
    out = res.json()
    assert "campaign_content" in out
    assert len(out["campaign_content"]) > 100


def test_api_migration():
    res = client.get("/api/migration")
    assert res.status_code == 200
    mig = res.json()
    assert mig.get("available") is True
    assert "flows" in mig and len(mig["flows"]) > 0
    assert "summary" in mig


def test_api_market_basket():
    res = client.get("/api/market-basket")
    assert res.status_code == 200
    mb = res.json()
    assert mb.get("available") is True
    assert "rules" in mb and len(mb["rules"]) > 0


def test_api_customer_detail_xai_and_recs():
    res = client.get("/api/customers/C00000001")
    assert res.status_code == 200
    detail = res.json()
    assert "explanation" in detail and len(detail["explanation"]) >= 2
    assert "recommendations" in detail and len(detail["recommendations"]) >= 1
    assert "segment_benchmark" in detail


def test_api_campaign_targets():
    res = client.get("/api/ai/campaign-targets", params={"segment_name": "Best Customers", "limit": 10})
    assert res.status_code == 200
    out = res.json()
    assert out["count"] == 10
    assert "Recommended_CrossSell" in out["records"][0]
    assert "Voucher_Code" in out["records"][0]

