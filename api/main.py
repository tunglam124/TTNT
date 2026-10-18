"""FastAPI Backend cho Hệ thống AI Phân tích & Giữ chân Khách hàng.

Cung cấp REST API đầy đủ:
- /api/overview: Số liệu tổng quan KPI, phân phối doanh thu
- /api/segments: Danh sách phân khúc, đặc điểm R-F-M, tỷ lệ
- /api/customers: Danh sách khách hàng (phân trang, lọc theo cụm, sắp xếp)
- /api/customers/{id}: Hồ sơ chi tiết Customer 360 & lịch sử giao dịch
- /api/score: Chấm điểm phân khúc & churn theo thời gian thực cho dữ liệu mới
- /api/train: Kích hoạt pipeline huấn luyện ngầm (Background Task)
- /api/runs: Quản lý các phiên huấn luyện trong Model Registry
- /api/ai/campaign: Trợ lý AI tạo chiến dịch Marketing cá nhân hóa
- /api/ai/ask: AI Copilot hỏi đáp insight kinh doanh
- /api/report/docx: Tải báo cáo Word (.docx) tự động
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Đảm bảo thư mục gốc dự án luôn có trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config.settings import settings
from src import pipeline, service
from src.reports.docx_generator import build_report

app = FastAPI(
    title="Customer Intelligence & Retention AI System API",
    description="REST API cho hệ thống phân tích phân khúc khách hàng RFM + K-Means + Predictive AI + Generative AI Copilot.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trạng thái tác vụ huấn luyện ngầm
_training_state = {
    "is_training": False,
    "progress": 0.0,
    "message": "Sẵn sàng",
    "last_run_id": None,
    "error": None,
}


# --- Pydantic Schemas ---
class CustomerScoreInput(BaseModel):
    Recency: float = Field(..., description="Số ngày kể từ lần mua gần nhất")
    Frequency: float = Field(..., description="Số lần/ngày mua")
    Monetary: float = Field(..., description="Tổng số tiền đã chi tiêu")
    Tenure: Optional[float] = None
    Lines: Optional[float] = None
    AvgOrderValue: Optional[float] = None
    AvgDaysBetween: Optional[float] = None


class BatchScoreRequest(BaseModel):
    records: List[CustomerScoreInput]


class CampaignRequest(BaseModel):
    segment_name: str
    goal: str = "Tăng tỷ lệ giữ chân và mua lặp lại"
    channel: str = "Email"
    budget: float = 10_000_000
    tone: str = "Thân thiện"


class InsightRequest(BaseModel):
    question: str


class TrainRequest(BaseModel):
    final_k: Optional[int] = None
    run_comparison: bool = True
    train_predictor: bool = True


# --- Endpoints ---
@app.get("/", tags=["Hệ thống"])
def root():
    return {
        "system": "Customer Intelligence & Retention AI System",
        "status": "online",
        "docs": "/docs",
        "version": "2.0.0",
        "llm_enabled": settings.llm_enabled,
    }


@app.get("/health", tags=["Hệ thống"])
def health():
    return {"status": "healthy"}


@app.get("/api/overview", tags=["Phân tích Tổng quan"])
def get_overview():
    try:
        return service.overview()
    except service.NoModelError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/segments", tags=["Phân khúc"])
def get_segments():
    try:
        return service.segment_profiles()
    except service.NoModelError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/migration", tags=["Phân khúc"])
def get_segment_migration():
    try:
        return service.segment_migration()
    except service.NoModelError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/market-basket", tags=["Gợi ý Bán chéo"])
def get_market_basket():
    try:
        return service.market_basket()
    except service.NoModelError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/customers", tags=["Khách hàng"])
def get_customers(
    segment: Optional[str] = Query(None, description="Lọc theo tên phân khúc"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("Monetary", description="Cột sắp xếp"),
    ascending: bool = Query(False),
):
    try:
        return service.list_customers(
            segment=segment, limit=limit, offset=offset, sort_by=sort_by, ascending=ascending
        )
    except service.NoModelError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/customers/{customer_id}", tags=["Khách hàng"])
def get_customer_detail(customer_id: str):
    try:
        return service.customer_detail(customer_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy khách hàng {customer_id}")
    except service.NoModelError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/score", tags=["Chấm điểm Online"])
def score_customers(req: BatchScoreRequest):
    try:
        data = [r.model_dump() for r in req.records]
        return service.score_rfm(data)
    except service.NoModelError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/runs", tags=["Model Registry"])
def list_runs():
    reg = service.registry()
    return {
        "latest_run": reg.latest_id(),
        "runs": reg.list_runs(),
    }


@app.post("/api/runs/{run_id}/activate", tags=["Model Registry"])
def activate_run(run_id: str):
    try:
        service.registry().set_latest(run_id)
        service.invalidate_cache()
        return {"status": "success", "active_run_id": run_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Huấn luyện ngầm ---
def _bg_train(source, final_k, compare, train_pred):
    global _training_state
    _training_state["is_training"] = True
    _training_state["error"] = None

    def on_prog(pct, msg):
        _training_state["progress"] = pct
        _training_state["message"] = msg

    try:
        art = pipeline.run_pipeline(
            source=source,
            final_k=final_k,
            compare=compare,
            train_predictor=train_pred,
            progress=on_prog,
        )
        service.invalidate_cache()
        _training_state["last_run_id"] = art.run_id
        _training_state["message"] = f"Hoàn tất: {art.run_id}"
    except Exception as e:
        _training_state["error"] = str(e)
        _training_state["message"] = f"Lỗi: {e}"
    finally:
        _training_state["is_training"] = False


@app.post("/api/train", tags=["Huấn luyện"])
def trigger_training(req: TrainRequest, bg_tasks: BackgroundTasks):
    global _training_state
    if _training_state["is_training"]:
        raise HTTPException(status_code=409, detail="Một tác vụ huấn luyện khác đang chạy.")

    bg_tasks.add_task(
        _bg_train,
        source=settings.data_path,
        final_k=req.final_k,
        compare=req.run_comparison,
        train_pred=req.train_predictor,
    )
    return {"status": "started", "message": "Quá trình huấn luyện đã được bắt đầu chạy ngầm."}


@app.get("/api/train/status", tags=["Huấn luyện"])
def get_training_status():
    return _training_state


# --- Generative AI & Copilot ---
@app.post("/api/ai/campaign", tags=["Generative AI"])
def generate_campaign(req: CampaignRequest):
    try:
        profiles = service.segment_profiles(for_llm=True)
        seg = next((p for p in profiles if p["Segment"] == req.segment_name), None)
        if not seg:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy phân khúc {req.segment_name}")
        tot = service.totals()
        copilot = service.copilot()
        res = copilot.generate_campaign(
            seg=seg, totals=tot, goal=req.goal, channel=req.channel,
            budget=req.budget, tone=req.tone
        )
        return {
            "engine": res.engine,
            "campaign_content": res.text,
            "error": res.error,
        }
    except service.NoModelError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/ai/ask", tags=["Generative AI"])
def ask_copilot(req: InsightRequest):
    try:
        profiles = service.segment_profiles(for_llm=True)
        tot = service.totals()
        copilot = service.copilot()
        res = copilot.answer_insight(profiles=profiles, totals=tot, question=req.question)
        return {
            "engine": res.engine,
            "answer": res.text,
            "error": res.error,
        }
    except service.NoModelError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/ai/campaign-targets", tags=["Generative AI"])
def get_campaign_targets(
    segment_name: str = Query(..., description="Tên phân khúc mục tiêu"),
    channel: str = Query("Email", description="Kênh chiến dịch"),
    limit: int = Query(500, ge=1, le=10000),
):
    try:
        df = service.export_campaign_targets(segment_name=segment_name, channel=channel, limit=limit)
        return {
            "segment_name": segment_name,
            "channel": channel,
            "count": int(len(df)),
            "records": df.to_dict("records"),
        }
    except service.NoModelError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/report/docx", tags=["Báo cáo"])
def download_docx_report():
    try:
        art = service.get_run()
        out_dir = settings.reports_dir
        path = build_report(art, out_dir)
        return FileResponse(
            path=str(path),
            filename=path.name,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    except service.NoModelError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tạo báo cáo: {e}")
