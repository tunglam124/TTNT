"""Prompt templates cho Marketing Copilot (dùng khi có Gemini API key)."""
from __future__ import annotations

SYSTEM_PROMPT = """Bạn là chuyên gia CRM & Marketing bán lẻ tại Việt Nam.
Bạn nhận số liệu phân khúc khách hàng thật (RFM, dự báo churn, giá trị dự báo) và phải:
- Chỉ dựa trên số liệu được cung cấp, không bịa thêm số liệu.
- Trả lời bằng tiếng Việt, định dạng Markdown, ngắn gọn, có thể hành động ngay.
- Nêu rõ KPI đo lường và giả định khi ước tính hiệu quả."""


def segment_context(seg: dict, totals: dict) -> str:
    lines = [
        f"Phân khúc: {seg['Segment']}",
        f"- Số khách: {seg['customers']:,} ({seg['customer_pct']:.1f}% tổng khách)",
        f"- Recency TB: {seg['recency_mean']:.0f} ngày | Frequency TB: {seg['frequency_mean']:.2f} | "
        f"Monetary TB: {seg['monetary_mean']:,.0f}",
        f"- Đóng góp doanh thu lịch sử: {seg['revenue_pct']:.1f}%",
    ]
    if seg.get("churn_prob_mean") is not None:
        lines.append(f"- Xác suất rời bỏ TB ({totals.get('horizon', 90)} ngày tới): {seg['churn_prob_mean']:.0%}")
    if seg.get("pred_value_90d_mean") is not None:
        lines.append(f"- Chi tiêu dự báo TB {totals.get('horizon', 90)} ngày tới: {seg['pred_value_90d_mean']:,.0f}")
    lines.append(f"Toàn hệ thống: {totals['customers']:,} khách, doanh thu {totals['revenue']:,.0f}.")
    return "\n".join(lines)


def campaign_prompt(context: str, goal: str, channel: str, budget: float, tone: str) -> str:
    return f"""{context}

Yêu cầu: Thiết kế chiến dịch marketing cho phân khúc trên.
- Mục tiêu: {goal}
- Kênh: {channel}
- Ngân sách: {budget:,.0f} VND
- Giọng văn: {tone}

Trả về đúng các mục sau:
## 1. Nhận định phân khúc
## 2. Chiến lược & ưu đãi (cụ thể mức giảm/quà, điều kiện)
## 3. Lịch triển khai (theo tuần)
## 4. Nội dung mẫu (tiêu đề + nội dung tin nhắn cho kênh đã chọn, có biến {{ten_khach}})
## 5. Phân bổ ngân sách & KPI đo lường (kèm cách A/B test)"""


def insight_prompt(context_all: str, question: str) -> str:
    return f"""Số liệu các phân khúc khách hàng:
{context_all}

Câu hỏi của quản lý: {question}
Trả lời ngắn gọn, dựa trên số liệu, kết thúc bằng 3 hành động ưu tiên."""


def customer_prompt(profile: str) -> str:
    return f"""Hồ sơ khách hàng:
{profile}

Đề xuất "Next Best Action" cho khách này: 1 hành động chính, 1 ưu đãi cụ thể, kênh liên hệ,
thời điểm liên hệ và 1 tin nhắn mẫu (tối đa 300 ký tự)."""
