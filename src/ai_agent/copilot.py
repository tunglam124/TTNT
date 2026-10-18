"""Marketing Copilot: dùng Gemini nếu có GEMINI_API_KEY, ngược lại dùng bộ sinh nội dung offline
dựa trên luật + số liệu thật của phân khúc (hệ thống luôn hoạt động kể cả khi không có mạng/API key)."""
from __future__ import annotations

from dataclasses import dataclass

from config.settings import settings
from src.models.segmentation import SEGMENT_INFO, base_segment

from . import prompts


@dataclass
class CopilotResponse:
    text: str
    engine: str  # "gemini" | "offline"
    error: str | None = None


class MarketingCopilot:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model = model or settings.gemini_model
        self._client = None

    @property
    def online(self) -> bool:
        return bool(self.api_key)

    def _llm(self, prompt: str) -> str:
        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        resp = self._client.models.generate_content(
            model=self.model, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=prompts.SYSTEM_PROMPT, temperature=0.7),
        )
        return resp.text or ""

    def _run(self, prompt: str, fallback) -> CopilotResponse:
        if self.online:
            try:
                return CopilotResponse(self._llm(prompt), "gemini")
            except Exception as e:  # lỗi mạng/quota -> vẫn trả kết quả offline
                return CopilotResponse(fallback(), "offline", error=str(e))
        return CopilotResponse(fallback(), "offline")

    # ------------------------------------------------------------------ public API
    def generate_campaign(self, seg: dict, totals: dict, goal: str, channel: str,
                          budget: float, tone: str = "Thân thiện") -> CopilotResponse:
        ctx = prompts.segment_context(seg, totals)
        return self._run(prompts.campaign_prompt(ctx, goal, channel, budget, tone),
                         lambda: offline_campaign(seg, totals, goal, channel, budget, tone))

    def answer_insight(self, profiles: list[dict], totals: dict, question: str) -> CopilotResponse:
        ctx = "\n\n".join(prompts.segment_context(p, totals) for p in profiles)
        return self._run(prompts.insight_prompt(ctx, question),
                         lambda: offline_insight(profiles, totals))

    def next_best_action(self, customer: dict) -> CopilotResponse:
        profile = "\n".join(f"- {k}: {v}" for k, v in customer.items())
        return self._run(prompts.customer_prompt(profile), lambda: offline_next_action(customer))


# ====================================================================== OFFLINE ENGINE
PLAYBOOK = {
    "Best Customers": {
        "insight": "Nhóm mang lại giá trị cao nhất; ưu tiên giữ chân và tăng gắn kết, KHÔNG giảm giá sâu.",
        "offer": "Đặc quyền VIP: miễn phí giao hàng 3 tháng + quà tặng sinh nhật + mua sớm sản phẩm mới",
        "discount": 0.05, "expected_response": 0.25,
    },
    "Loyal Customers": {
        "insight": "Khách ổn định; cơ hội tăng giá trị đơn qua bán chéo/bán thêm.",
        "offer": "Tích điểm x2 cho đơn từ {aov} trở lên + combo bán chéo giảm 10%",
        "discount": 0.10, "expected_response": 0.18,
    },
    "Potential Customers": {
        "insight": "Mới mua gần đây nhưng tần suất thấp; cần tạo thói quen mua lần 2.",
        "offer": "Voucher 15% cho lần mua tiếp theo trong 14 ngày",
        "discount": 0.15, "expected_response": 0.12,
    },
    "At-Risk Customers": {
        "insight": "Từng chi tiêu tốt nhưng lâu không quay lại; cần hành động sớm trước khi mất hẳn.",
        "offer": "Ưu đãi 'Chào mừng trở lại' 20% (tối đa {cap}) hạn 10 ngày + khảo sát 1 câu",
        "discount": 0.20, "expected_response": 0.08,
    },
    "Lost Customers": {
        "insight": "Giá trị thấp và đã rời đi lâu; chỉ đầu tư chi phí thấp, chấp nhận tỷ lệ phản hồi thấp.",
        "offer": "Voucher 25% có điều kiện đơn tối thiểu + khảo sát lý do rời bỏ",
        "discount": 0.25, "expected_response": 0.03,
    },
}
CHANNEL_COST = {"Email": 500, "SMS": 800, "Zalo OA": 600, "Push notification": 100, "Facebook Ads": 3000}


def _play(seg_name: str) -> dict:
    return PLAYBOOK.get(base_segment(seg_name), PLAYBOOK["Potential Customers"])


def _fmt(x: float) -> str:
    return f"{x:,.0f}"


def offline_campaign(seg: dict, totals: dict, goal: str, channel: str, budget: float, tone: str) -> str:
    name = seg["Segment"]
    play = _play(name)
    n = int(seg["customers"])
    aov = seg["monetary_mean"] / max(seg["frequency_mean"], 1)
    offer = play["offer"].format(aov=_fmt(aov * 1.2), cap=_fmt(aov * 0.3))
    cost_per = CHANNEL_COST.get(channel, 1000)
    reach = int(min(n, budget * 0.4 / cost_per)) if budget > 0 else n
    resp = play["expected_response"]
    if seg.get("churn_prob_mean") is not None:
        resp *= (1.2 - 0.4 * seg["churn_prob_mean"])  # khách churn cao phản hồi kém hơn
    buyers = int(reach * resp)
    revenue = buyers * aov
    voucher_cost = revenue * play["discount"]
    greeting = {"Trang trọng": "Kính gửi Quý khách {ten_khach},", "Hài hước": "Ê {ten_khach} ơi! 👋",
                }.get(tone, "Chào {ten_khach} 👋,")

    churn_line = ""
    if seg.get("churn_prob_mean") is not None:
        churn_line = (f"- Mô hình dự báo **{seg['churn_prob_mean']:.0%}** khách trong nhóm sẽ không mua trong "
                      f"{totals.get('horizon', 90)} ngày tới nếu không có tác động.\n")

    cat_line = ""
    if seg.get("top_category"):
        cat_line = f"- Ngành hàng chủ lực của nhóm: **{seg['top_category']}**"
        if seg.get("preferred_channel"):
            cat_line += f" | Kênh mua sắm ưa thích: **{seg['preferred_channel']}**"
        cat_line += ".\n"

    cross_sell_hint = (
        f"gợi ý mua kèm ngành hàng **{seg['top_category']}** và các sản phẩm liên kết có Lift cao"
        if seg.get("top_category")
        else "gợi ý nhóm sản phẩm khách đã mua nhiều nhất"
    )

    return f"""## 1. Nhận định phân khúc — {name}
- {SEGMENT_INFO.get(base_segment(name), '')}
- Quy mô: **{n:,} khách** ({seg['customer_pct']:.1f}%), đóng góp **{seg['revenue_pct']:.1f}%** doanh thu lịch sử.
- Lần mua gần nhất trung bình **{seg['recency_mean']:.0f} ngày** trước; tần suất **{seg['frequency_mean']:.2f}**; chi tiêu TB **{_fmt(seg['monetary_mean'])}**.
{cat_line}{churn_line}- Nhận định: {play['insight']}

## 2. Chiến lược & ưu đãi
- **Mục tiêu:** {goal}
- **Ưu đãi chính:** {offer}
- **Kênh:** {channel} (chi phí ước tính {_fmt(cost_per)}/lượt gửi)
- **Cá nhân hóa giỏ hàng (Next-Best-Category):** {cross_sell_hint}.

## 3. Lịch triển khai
| Tuần | Hoạt động |
|---|---|
| 1 | Chọn nhóm thử nghiệm A/B (10% khách làm nhóm đối chứng không nhận ưu đãi), soạn nội dung |
| 2 | Gửi đợt 1 qua {channel} |
| 3 | Nhắc lại cho người chưa phản hồi (đổi tiêu đề/thời điểm gửi) |
| 4 | Đo lường, so sánh với nhóm đối chứng, quyết định mở rộng |

## 4. Nội dung mẫu ({channel}, giọng văn: {tone})
**Tiêu đề:** {"Đặc quyền dành riêng cho bạn" if "Best" in name else "Có món quà đang chờ bạn"}

> {greeting}
> Cảm ơn bạn đã đồng hành cùng chúng tôi. Dành riêng cho bạn: **{offer}**.
> Ưu đãi có hạn — mua ngay tại cửa hàng hoặc ứng dụng!

## 5. Ngân sách & KPI (ước tính sơ bộ)
| Hạng mục | Giá trị |
|---|---|
| Ngân sách | {_fmt(budget)} VND |
| Số khách tiếp cận (40% ngân sách cho kênh) | {reach:,} |
| Tỷ lệ phản hồi giả định | {resp:.1%} |
| Số khách mua dự kiến | {buyers:,} |
| Doanh thu dự kiến (theo giá trị đơn TB {_fmt(aov)}) | {_fmt(revenue)} |
| Chi phí ưu đãi ({play['discount']:.0%}) | {_fmt(voucher_cost)} |

**KPI:** tỷ lệ mở/nhấp, tỷ lệ chuyển đổi so với nhóm đối chứng, doanh thu tăng thêm (uplift), tỷ lệ quay lại sau 30/60 ngày.

> _Nội dung được tạo bởi bộ sinh offline dựa trên số liệu phân khúc. Thêm `GEMINI_API_KEY` vào file `.env` để dùng Gemini sinh nội dung linh hoạt hơn._
"""


def offline_insight(profiles: list[dict], totals: dict) -> str:
    by_rev = sorted(profiles, key=lambda p: -p["revenue_pct"])
    top = by_rev[0]
    lines = ["## Tổng quan nhanh",
             f"- Hệ thống có **{totals['customers']:,} khách**, doanh thu lịch sử **{_fmt(totals['revenue'])}**.",
             f"- **{top['Segment']}** đóng góp nhiều doanh thu nhất (**{top['revenue_pct']:.1f}%**) "
             f"với {top['customer_pct']:.1f}% số khách."]
    risky = [p for p in profiles if base_segment(p["Segment"]) == "At-Risk Customers"]
    for p in risky:
        lines.append(f"- ⚠️ **{p['Segment']}**: {p['customers']:,} khách, {p['revenue_pct']:.1f}% doanh thu, "
                     f"đã {p['recency_mean']:.0f} ngày chưa mua — rủi ro mất doanh thu lớn nhất.")
    if any(p.get("churn_prob_mean") is not None for p in profiles):
        worst = max(profiles, key=lambda p: p.get("churn_prob_mean") or 0)
        lines.append(f"- Nhóm có xác suất rời bỏ cao nhất: **{worst['Segment']}** ({worst['churn_prob_mean']:.0%}).")
    lines += ["", "## 3 hành động ưu tiên",
              "1. Chạy chiến dịch tái kích hoạt cho nhóm At-Risk trong 2 tuần tới (có nhóm đối chứng).",
              "2. Chương trình đặc quyền cho nhóm giá trị cao nhất để giữ chân, tránh giảm giá sâu.",
              "3. Hạn chế ngân sách cho nhóm Lost; dùng kênh chi phí thấp (email/push) + khảo sát lý do.",
              "", "> _Trả lời bởi bộ phân tích offline. Thêm GEMINI_API_KEY để hỏi đáp tự do bằng ngôn ngữ tự nhiên._"]
    return "\n".join(lines)


def offline_next_action(c: dict) -> str:
    play = _play(str(c.get("Segment", "")))
    churn = c.get("churn_prob")
    urgency = "trong 3 ngày tới" if churn is not None and churn >= 0.7 else "trong 2 tuần tới"
    pref_ch = c.get("PreferredChannel")
    channel = f"SMS/Zalo ({pref_ch})" if pref_ch and churn is not None and churn >= 0.7 else (pref_ch or ("SMS/Zalo" if churn is not None and churn >= 0.7 else "Email"))
    aov = float(c.get("AvgOrderValue", c.get("Monetary", 0)) or 0)
    offer = play["offer"].format(aov=_fmt(aov * 1.2), cap=_fmt(aov * 0.3))
    top_cat = c.get("TopCategory")
    cat_note = f"\n- **Gợi ý ngành hàng ưu tiên:** {top_cat}" if top_cat else ""
    return f"""**Hành động chính:** {play['insight']}
- **Ưu đãi:** {offer}
- **Kênh:** {channel} — liên hệ {urgency}{cat_note}
- **Tin nhắn mẫu:** "Chào bạn, đã {c.get('Recency', '?')} ngày rồi chúng tôi chưa gặp bạn! Tặng bạn ưu đãi: {offer}. Hẹn gặp lại bạn sớm!"
"""
