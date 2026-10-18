"""Streamlit Web Dashboard: Hệ thống AI Phân tích & Giữ chân Khách hàng Toàn diện (v2.5).

Giao diện tương tác trực quan 5 chức năng chính + gói nâng cấp AI nâng cao:
1. 📊 Tổng quan Kinh doanh, KPI & Dòng dịch chuyển phân khúc (Sankey Migration)
2. 🎯 Không gian Phân cụm RFM 3D, Biểu đồ Radar 5 chiều & Luật kết hợp Giỏ hàng (Market Basket)
3. 👤 Hồ sơ Khách hàng 360, Explainable AI, Gợi ý Bán chéo & Mô phỏng What-If
4. 🤖 AI Marketing Studio (GenAI Copilot) & Xuất Danh sách Khách hàng Mục tiêu (.CSV)
5. ⚙️ Quản lý Mô hình, Huấn luyện & Xuất Báo cáo Word (.DOCX)
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# Đảm bảo thư mục gốc dự án luôn có trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import settings
from src import pipeline, service
from src.reports.docx_generator import build_report

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="AI Customer Intelligence & Retention System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho giao diện hiện đại, chuyên nghiệp
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .explain-box {
        background: #EFF6FF;
        border-left: 4px solid #2563EB;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 1rem;
    }
    .rec-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px;
        height: 100%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def load_overview_data():
    return service.overview()


@st.cache_data(ttl=60)
def load_segment_profiles():
    return service.segment_profiles()


@st.cache_data(ttl=60)
def load_segment_migration():
    return service.segment_migration()


@st.cache_data(ttl=60)
def load_market_basket():
    return service.market_basket()


def get_current_run():
    return service.get_run()


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullseye.png", width=64)
    st.title("Customer AI v2.5")
    st.caption("RFM+ · Explainable AI · Cross-Sell · What-If · Copilot")

    try:
        current_art = get_current_run()
        run_meta = current_art.metadata
        st.success(f"Phiên hiện tại: **{current_art.run_id}**")
        st.write(f"📅 **Mốc dữ liệu:** `{run_meta['snapshot_date']}`")
        st.write(f"👥 **Số khách hàng:** `{len(current_art.customers):,}`")
        st.write(f"🎯 **Số cụm ($k$):** `{run_meta['k']}`")
    except Exception:
        st.warning("Chưa có phiên mô hình nào hoạt động.")

    st.markdown("---")
    menu = st.radio(
        "Điều hướng hệ thống:",
        [
            "1. 📊 Tổng quan & Dịch chuyển",
            "2. 🎯 Phân cụm 3D, Radar & Giỏ hàng",
            "3. 👤 Hồ sơ 360, XAI & What-If",
            "4. 🤖 AI Marketing Studio",
            "5. ⚙️ Quản lý Mô hình & Dữ liệu",
        ],
        index=0,
    )

    st.markdown("---")
    copilot_mode = "🟢 Gemini API (Online)" if settings.llm_enabled else "🟡 Rule-based Copilot (Offline)"
    st.caption(f"Trợ lý AI: {copilot_mode}")
    if not settings.llm_enabled:
        st.caption("*(Thêm `GEMINI_API_KEY` vào file `.env` để kích hoạt Gemini LLM)*")


# ==============================================================================
# TAB 1: TỔNG QUAN KINH DOANH, KPI & DÒNG DỊCH CHUYỂN PHÂN KHÚC (SANKEY)
# ==============================================================================
if menu.startswith("1."):
    st.markdown('<div class="main-header">📊 Tổng quan Kinh doanh & Dịch chuyển Phân khúc</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Báo cáo phân bổ doanh thu, đặc trưng RFM+ và ma trận dịch chuyển khách hàng qua thời gian</div>', unsafe_allow_html=True)

    try:
        ov = load_overview_data()
        profiles = load_segment_profiles()
        prof_df = pd.DataFrame(profiles)

        # Hàng KPI chính
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Tổng khách hàng", f"{ov['customers']:,}")
        with c2:
            st.metric("Tổng doanh thu (Monetary)", f"{ov['revenue']:,.0f}")
        with c3:
            st.metric("Chi tiêu TB / khách", f"{ov['avg_monetary']:,.1f}")
        with c4:
            st.metric("Tần suất mua TB", f"{ov['avg_frequency']:.2f} lần")
        with c5:
            st.metric("Tỷ lệ Active 90 ngày", f"{ov['active_90d_pct']:.1f}%")

        st.markdown("---")

        col_left, col_right = st.columns([6, 4])
        with col_left:
            st.subheader("Doanh thu đóng góp theo Phân khúc")
            fig_bar = px.bar(
                prof_df,
                x="Segment",
                y="revenue_pct",
                color="Segment",
                text=prof_df["revenue_pct"].apply(lambda x: f"{x:.1f}%"),
                labels={"revenue_pct": "% Doanh thu", "Segment": "Phân khúc"},
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )
            fig_bar.update_layout(showlegend=False, height=360, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_bar, width="stretch")

        with col_right:
            st.subheader("Tỷ lệ Số lượng Khách hàng")
            fig_pie = px.pie(
                prof_df,
                names="Segment",
                values="customer_pct",
                hole=0.45,
                color="Segment",
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(showlegend=False, height=360, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, width="stretch")

        st.subheader("Bảng Chỉ số Chi tiết từng Phân khúc (RFM+)")
        display_cols = {
            "Segment": "Tên Phân khúc",
            "customers": "Số khách hàng",
            "customer_pct": "% Khách hàng",
            "revenue_pct": "% Doanh thu",
            "recency_mean": "Recency TB (ngày)",
            "frequency_mean": "Frequency TB",
            "monetary_mean": "Monetary TB",
            "aov_mean": "AOV TB",
            "category_count_mean": "Số ngành hàng TB",
            "top_category": "Ngành hàng chủ lực",
            "preferred_channel": "Kênh ưa thích",
            "profit_pct": "% Lợi nhuận gộp",
        }
        df_show = prof_df[[c for c in display_cols.keys() if c in prof_df.columns]].rename(columns=display_cols)
        fmt_map = {
            "Số khách hàng": "{:,}",
            "% Khách hàng": "{:.1f}%",
            "% Doanh thu": "{:.1f}%",
            "Recency TB (ngày)": "{:.1f}",
            "Frequency TB": "{:.2f}",
            "Monetary TB": "{:,.1f}",
            "AOV TB": "{:,.1f}",
            "Số ngành hàng TB": "{:.1f}",
            "% Lợi nhuận gộp": "{:.1f}%",
        }
        active_fmt = {k: v for k, v in fmt_map.items() if k in df_show.columns}
        st.dataframe(df_show.style.format(active_fmt, na_rep="—"), width="stretch", hide_index=True)

        # --- PHÂN TÍCH DỊCH CHUYỂN PHÂN KHÚC (SEGMENT MIGRATION SANKEY) ---
        st.markdown("---")
        st.subheader("🔄 Phân tích Dịch chuyển Phân khúc Khách hàng (90 ngày trước ➔ Hiện tại)")
        mig = load_segment_migration()
        if mig and mig.get("available") and mig.get("flows"):
            msum = mig.get("summary", {})
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("Giữ nguyên phân khúc", f"{msum.get('retained_same_segment', 0):,} khách")
            with mc2:
                st.metric("Khách hàng mới gia nhập", f"{msum.get('new_customers', 0):,} khách")
            with mc3:
                st.metric(
                    "⚠️ Trượt xuống At-Risk / Lost",
                    f"{msum.get('slipped_to_risk_or_lost', 0):,} khách",
                    delta="-Cần giữ chân",
                    delta_color="inverse",
                )
            with mc4:
                st.metric(
                    "🎉 Tái kích hoạt (Won-back)",
                    f"{msum.get('reactivated_won_back', 0):,} khách",
                    delta="+Tích cực",
                )

            st.caption(
                f"So sánh trạng thái phân khúc của cùng tập khách hàng tại mốc cắt **{mig.get('cutoff_date')}** "
                f"so với mốc hiện tại **{mig.get('snapshot_date')}** (cửa sổ {mig.get('horizon_days', 90)} ngày)."
            )

            flows = mig["flows"]
            src_labels = sorted({f["source"] for f in flows})
            tgt_labels = sorted({f["target"] for f in flows})
            all_nodes = [f"Trước: {s}" for s in src_labels] + [f"Nay: {t}" for t in tgt_labels]
            src_idx_map = {s: i for i, s in enumerate(src_labels)}
            tgt_idx_map = {t: i + len(src_labels) for i, t in enumerate(tgt_labels)}

            fig_sankey = go.Figure(
                data=[
                    go.Sankey(
                        node=dict(
                            pad=18,
                            thickness=22,
                            line=dict(color="#CBD5E1", width=1),
                            label=all_nodes,
                        ),
                        link=dict(
                            source=[src_idx_map[f["source"]] for f in flows],
                            target=[tgt_idx_map[f["target"]] for f in flows],
                            value=[f["count"] for f in flows],
                        ),
                    )
                ]
            )
            fig_sankey.update_layout(height=450, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_sankey, width="stretch")
        else:
            st.info("Dữ liệu dịch chuyển phân khúc sẽ khả dụng sau khi chạy lại Pipeline huấn luyện.")

    except Exception as e:
        st.error(f"Lỗi tải dữ liệu tổng quan: {e}")


# ==============================================================================
# TAB 2: KHÔNG GIAN PHÂN CỤM RFM 3D, RADAR 5 CHIỀU & GIỎ HÀNG
# ==============================================================================
elif menu.startswith("2."):
    st.markdown('<div class="main-header">🎯 Không gian Phân cụm 3D, Biểu đồ Radar & Giỏ hàng</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Trực quan hóa đa chiều R-F-M, chân dung 5 trục chuẩn hóa và luật kết hợp mua chéo (Market Basket)</div>', unsafe_allow_html=True)

    try:
        art = get_current_run()
        cust_df = art.customers
        meta = art.metadata
        profiles = load_segment_profiles()

        col_3d, col_radar = st.columns([6, 4])

        with col_3d:
            sample_size = min(4000, len(cust_df))
            plot_sample = cust_df.sample(sample_size, random_state=42)
            st.subheader(f"Không gian Phân cụm 3D R × F × M ({sample_size:,} khách)")
            fig_3d = px.scatter_3d(
                plot_sample,
                x="Recency",
                y="Frequency",
                z="Monetary",
                color="Segment",
                hover_name="CUSTOMER_ID",
                opacity=0.7,
                size_max=8,
                labels={"Recency": "Recency (ngày)", "Frequency": "Tần suất", "Monetary": "Tổng chi tiêu"},
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )
            fig_3d.update_layout(height=520, margin=dict(l=0, r=0, b=0, t=20))
            st.plotly_chart(fig_3d, width="stretch")

        with col_radar:
            st.subheader("🕸️ Biểu đồ Radar Chân dung 5 Chiều")
            st.caption("Điểm chuẩn hóa (0–100) của từng phân khúc trên 5 trục hành vi:")
            radar_axes = ["Độ mới (Recency)", "Tần suất (Frequency)", "Chi tiêu (Monetary)", "Giá trị đơn (AOV)", "Đa dạng ngành hàng"]
            fig_radar = go.Figure()
            for p in profiles:
                vals = [
                    float(p.get("radar_recency", 50)),
                    float(p.get("radar_frequency", 50)),
                    float(p.get("radar_monetary", 50)),
                    float(p.get("radar_aov", 50)),
                    float(p.get("radar_diversity", 50)),
                ]
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=radar_axes + [radar_axes[0]],
                        fill="toself",
                        name=p["Segment"],
                        opacity=0.65,
                    )
                )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                height=520,
                margin=dict(l=35, r=35, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25),
            )
            st.plotly_chart(fig_radar, width="stretch")

        # --- MARKET BASKET ASSOCIATION RULES ---
        st.markdown("---")
        st.subheader("🛒 Phân tích Giỏ hàng & Luật Kết hợp Bán chéo (Market Basket Association Rules)")
        mb = load_market_basket()
        if mb and mb.get("rules"):
            rules_df = pd.DataFrame(mb["rules"]).rename(
                columns={
                    "antecedent": "Khi khách mua ngành hàng (A)",
                    "consequent": "Gợi ý mua kèm ngành hàng (B)",
                    "pair_customers": "Số khách mua chung",
                    "support": "Support",
                    "confidence": "Confidence (Xác suất mua B khi đã mua A)",
                    "lift": "Lift (Độ nâng liên kết)",
                    "top_products": "Sản phẩm tiêu biểu của B",
                }
            )
            if "Sản phẩm tiêu biểu của B" in rules_df.columns:
                rules_df["Sản phẩm tiêu biểu của B"] = rules_df["Sản phẩm tiêu biểu của B"].apply(
                    lambda x: ", ".join(x) if isinstance(x, list) else str(x)
                )
            rule_fmt = {
                "Số khách mua chung": "{:,}",
                "Support": "{:.1%}",
                "Confidence (Xác suất mua B khi đã mua A)": "{:.1%}",
                "Lift (Độ nâng liên kết)": "{:.2f}x",
            }
            active_rule_fmt = {k: v for k, v in rule_fmt.items() if k in rules_df.columns}
            st.dataframe(
                rules_df.style.format(active_rule_fmt),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Chưa có dữ liệu luật kết hợp ngành hàng.")

        st.markdown("---")
        st.subheader("Đối sánh Định lượng 6 Thuật toán Phân cụm (Thẩm định JABES)")
        st.caption("Tất cả thuật toán được chấm trên CÙNG một mẫu chuẩn hóa để so sánh công bằng.")

        if meta.get("algorithm_comparison"):
            comp_df = pd.DataFrame(meta["algorithm_comparison"])
            st.dataframe(
                comp_df.style.format({
                    "n_clusters": "{:.0f}",
                    "noise_pct": "{:.1f}%",
                    "silhouette": "{:.3f}",
                    "davies_bouldin": "{:.3f}",
                    "calinski_harabasz": "{:,.1f}",
                    "dunn": "{:.3f}",
                }, na_rep="—"),
                width="stretch",
                hide_index=True,
            )

    except Exception as e:
        st.error(f"Lỗi trực quan hóa: {e}")


# ==============================================================================
# TAB 3: HỒ SƠ KHÁCH HÀNG 360, EXPLAINABLE AI, BÁN CHÉO & MÔ PHỎNG WHAT-IF
# ==============================================================================
elif menu.startswith("3."):
    st.markdown('<div class="main-header">👤 Hồ sơ Khách hàng 360, Giải thích AI & Mô phỏng What-If</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Tra cứu hành vi cá nhân, giải thích lý do phân cụm (XAI), gợi ý bán chéo và mô phỏng kịch bản</div>', unsafe_allow_html=True)

    try:
        art = get_current_run()
        cust_df = art.customers

        col_search1, col_search2 = st.columns([4, 2])
        default_cid = str(cust_df["CUSTOMER_ID"].iloc[0]) if len(cust_df) > 0 else ""
        with col_search1:
            search_id = st.text_input("Nhập Mã Khách Hàng (CUSTOMER_ID):", value=default_cid).strip()
        with col_search2:
            sample_ids = cust_df["CUSTOMER_ID"].head(15).tolist()
            chosen_sample = st.selectbox("Hoặc chọn mẫu nhanh:", ["-- Chọn mẫu --"] + sample_ids)
            if chosen_sample != "-- Chọn mẫu --":
                search_id = chosen_sample

        if search_id:
            try:
                detail = service.customer_detail(search_id)
                c_data = detail["customer"]
                pct = detail["percentiles"]
                txs = detail["transactions"]
                expl = detail.get("explanation", [])
                recs = detail.get("recommendations", [])
                bought_cats = detail.get("purchased_categories", [])
                seg_bench = detail.get("segment_benchmark", {})

                st.markdown(f"### Khách hàng: `{search_id}` — Phân khúc: **{c_data['Segment']}**")

                kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                with kpi1:
                    st.metric("Recency", f"{c_data['Recency']} ngày", f"Top {pct['Recency']:.0f}% gần đây")
                with kpi2:
                    st.metric("Frequency", f"{c_data['Frequency']} đơn", f"Top {pct['Frequency']:.0f}%")
                with kpi3:
                    st.metric("Monetary", f"{c_data['Monetary']:,.1f}", f"Top {pct['Monetary']:.0f}% chi tiêu")
                with kpi4:
                    st.metric("Giá trị đơn TB (AOV)", f"{c_data.get('AvgOrderValue', 0):,.1f}", f"Top {pct.get('AvgOrderValue', 50):.0f}%")
                with kpi5:
                    churn_val = c_data.get("churn_prob")
                    st.metric(
                        "Xác suất Rời bỏ (Churn)",
                        f"{churn_val:.1%}" if churn_val is not None else "—",
                        f"Kênh: {c_data.get('PreferredChannel', '—')}",
                    )

                # --- EXPLAINABLE AI & INDIVIDUAL RADAR CHART ---
                col_xai, col_cradar = st.columns([6, 4])
                with col_xai:
                    st.subheader("🧠 Giải thích Lý do Phân cụm (Explainable AI)")
                    if expl:
                        reasons_html = "".join(f"<li style='margin-bottom:6px;'>{r}</li>" for r in expl)
                        st.markdown(
                            f"<div class='explain-box'><b>Tại sao khách hàng <code>{search_id}</code> thuộc nhóm "
                            f"<code>{c_data['Segment']}</code>?</b><ul style='margin-top:8px;margin-bottom:0;'>{reasons_html}</ul></div>",
                            unsafe_allow_html=True,
                        )
                    st.subheader("💡 Khuyến nghị Hành động Cá nhân hóa (Next Best Action)")
                    copilot = service.copilot()
                    act_resp = copilot.next_best_action(c_data)
                    st.info(act_resp.text)

                with col_cradar:
                    st.subheader("🕸️ So sánh Cá nhân vs. Chuẩn Phân khúc")
                    radar_axes = ["Độ mới (Recency)", "Tần suất", "Chi tiêu", "AOV", "Đa dạng ngành"]
                    c_vals = [
                        float(pct.get("Recency", 50)),
                        float(pct.get("Frequency", 50)),
                        float(pct.get("Monetary", 50)),
                        float(pct.get("AvgOrderValue", 50)),
                        float(pct.get("CategoryCount", 50)),
                    ]
                    s_vals = [
                        float(seg_bench.get("radar_recency", 50)),
                        float(seg_bench.get("radar_frequency", 50)),
                        float(seg_bench.get("radar_monetary", 50)),
                        float(seg_bench.get("radar_aov", 50)),
                        float(seg_bench.get("radar_diversity", 50)),
                    ]
                    fig_cr = go.Figure()
                    fig_cr.add_trace(
                        go.Scatterpolar(
                            r=c_vals + [c_vals[0]],
                            theta=radar_axes + [radar_axes[0]],
                            fill="toself",
                            name=f"Khách {search_id}",
                        )
                    )
                    fig_cr.add_trace(
                        go.Scatterpolar(
                            r=s_vals + [s_vals[0]],
                            theta=radar_axes + [radar_axes[0]],
                            fill="toself",
                            name=f"TB {c_data['Segment']}",
                        )
                    )
                    fig_cr.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        height=360,
                        margin=dict(l=30, r=30, t=20, b=20),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                    )
                    st.plotly_chart(fig_cr, width="stretch")

                # --- NEXT-BEST-CATEGORY CROSS-SELL RECOMMENDATIONS ---
                st.markdown("---")
                st.subheader("🛍️ Gợi ý Bán chéo Ngành hàng Tiếp theo (Next-Best-Category Recommender)")
                if bought_cats:
                    st.caption(f"**Ngành hàng đã mua ({len(bought_cats)}):** " + ", ".join(f"`{c}`" for c in bought_cats))
                if recs:
                    rc_cols = st.columns(len(recs))
                    for idx, r in enumerate(recs):
                        with rc_cols[idx]:
                            prods = ", ".join(r.get("sample_products", [])) or "Các sản phẩm nổi bật"
                            lift_str = f"Lift: <b>{r['lift']:.2f}x</b> · Confidence: <b>{r['confidence']:.0%}</b>" if r.get("confidence", 0) > 0 else "Phổ biến trong phân khúc"
                            st.markdown(
                                f"<div class='rec-card'>"
                                f"<div style='font-size:0.8rem;color:#64748B;'>Gợi ý #{idx+1}</div>"
                                f"<div style='font-size:1.2rem;font-weight:700;color:#1E3A8A;margin:4px 0;'>🎯 {r['recommended_category']}</div>"
                                f"<div style='font-size:0.85rem;color:#047857;margin-bottom:6px;'>{lift_str}</div>"
                                f"<div style='font-size:0.88rem;color:#334155;'>{r['reason']}</div>"
                                f"<div style='font-size:0.82rem;color:#475569;margin-top:8px;'><b>Sản phẩm tiêu biểu:</b> {prods}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.caption("Chưa có gợi ý bán chéo cho khách hàng này.")

                # --- INTERACTIVE WHAT-IF SIMULATOR ---
                st.markdown("---")
                st.subheader("🎛️ Bộ Mô phỏng Kịch bản Hành vi 'What-If' (Interactive Simulator)")
                st.caption("Kéo thanh trượt để xem: Nếu khách hàng này mua thêm đơn hoặc quay lại sớm hơn, họ sẽ chuyển sang phân khúc nào và rủi ro rời bỏ giảm bao nhiêu?")

                sim_c1, sim_c2, sim_c3, sim_c4 = st.columns([3, 3, 3, 4])
                max_r = int(max(365, cust_df["Recency"].max()))
                max_f = int(max(30, cust_df["Frequency"].quantile(0.99) * 1.5))
                max_m = float(max(5000.0, cust_df["Monetary"].quantile(0.99) * 1.5))

                with sim_c1:
                    sim_r = st.slider("Giả định Recency (ngày):", min_value=0, max_value=max_r, value=int(c_data["Recency"]))
                with sim_c2:
                    sim_f = st.slider("Giả định Frequency (số đơn):", min_value=1, max_value=max_f, value=max(1, int(c_data["Frequency"])))
                with sim_c3:
                    sim_m = st.slider(
                        "Giả định Monetary (tổng chi tiêu):",
                        min_value=10.0,
                        max_value=round(max_m, -1),
                        value=float(min(round(max_m, -1), max(10.0, float(c_data["Monetary"])))),
                        step=10.0,
                    )

                sim_res = service.score_rfm(
                    [
                        {
                            "Recency": sim_r,
                            "Frequency": sim_f,
                            "Monetary": sim_m,
                            "Tenure": max(float(c_data.get("Tenure", sim_r)), float(sim_r)),
                            "Lines": float(sim_f * 2),
                            "AvgOrderValue": float(sim_m / max(sim_f, 1)),
                            "AvgDaysBetween": float(c_data.get("AvgDaysBetween", 30.0)),
                        }
                    ]
                )[0]

                with sim_c4:
                    new_seg = sim_res["Segment"]
                    old_seg = c_data["Segment"]
                    seg_changed = new_seg != old_seg
                    st.metric(
                        "Phân khúc sau mô phỏng",
                        new_seg,
                        delta=f"Chuyển từ {old_seg}" if seg_changed else "Giữ nguyên phân khúc",
                        delta_color="normal" if seg_changed else "off",
                    )
                    if "churn_prob" in sim_res and c_data.get("churn_prob") is not None:
                        diff_churn = (sim_res["churn_prob"] - float(c_data["churn_prob"])) * 100
                        st.metric(
                            "Xác suất rời bỏ (mô phỏng)",
                            f"{sim_res['churn_prob']:.1%}",
                            delta=f"{diff_churn:+.1f}% điểm",
                            delta_color="inverse",
                        )

                # Lịch sử giao dịch
                st.markdown("---")
                st.subheader(f"🧾 Lịch sử Giao dịch ({len(txs)} dòng hóa đơn)")
                if txs:
                    st.dataframe(pd.DataFrame(txs), width="stretch", hide_index=True)
                else:
                    st.caption("Không có dữ liệu dòng hóa đơn chi tiết.")

            except KeyError:
                st.error(f"Không tìm thấy khách hàng `{search_id}` trong hệ thống.")
            except Exception as e:
                st.error(f"Lỗi truy xuất khách hàng: {e}")

    except Exception as e:
        st.error(f"Lỗi: {e}")


# ==============================================================================
# TAB 4: AI MARKETING STUDIO (COPILOT) & EXPORT DANH SÁCH MỤC TIÊU
# ==============================================================================
elif menu.startswith("4."):
    st.markdown('<div class="main-header">🤖 AI Marketing Studio & Xuất Danh sách Chiến dịch</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Trợ lý AI tự động lập chiến dịch, tối ưu ngân sách, cá nhân hóa thông điệp và xuất danh sách khách hàng mục tiêu (.CSV)</div>', unsafe_allow_html=True)

    sub_tab1, sub_tab2 = st.tabs(["🎯 Lập Chiến dịch & Xuất Danh sách Mục tiêu", "💬 Trò chuyện cùng AI Copilot"])

    with sub_tab1:
        try:
            profiles = load_segment_profiles()
            seg_names = [p["Segment"] for p in profiles]

            c1, c2 = st.columns(2)
            with c1:
                sel_seg = st.selectbox("Phân khúc mục tiêu:", seg_names, index=0)
                goal = st.selectbox(
                    "Mục tiêu chiến dịch:",
                    [
                        "Tái kích hoạt khách hàng đã lâu không quay lại (Win-back)",
                        "Tri ân và nâng cao lòng trung thành (VIP Retention)",
                        "Bán chéo / Tăng giá trị đơn hàng (Cross-sell / Upsell)",
                        "Khuyến khích mua lần 2 cho khách mới (2nd Purchase Nudge)",
                    ],
                )
            with c2:
                channel = st.selectbox("Kênh tiếp cận chính:", ["Email", "Zalo OA", "SMS", "Push notification", "Facebook Ads"])
                budget = st.number_input("Ngân sách dự kiến (VND):", min_value=500_000, max_value=500_000_000, value=10_000_000, step=1_000_000)
                tone = st.selectbox("Giọng văn thông điệp:", ["Thân thiện, trẻ trung", "Trang trọng, tri ân", "Gấp gáp, ưu đãi có hạn", "Hài hước, gần gũi"])

            if st.button("🚀 Sinh Chiến dịch bằng AI", type="primary"):
                with st.spinner("AI Copilot đang phân tích số liệu phân khúc và soạn thảo chiến dịch..."):
                    seg_data = next(p for p in profiles if p["Segment"] == sel_seg)
                    tot_data = service.totals()
                    copilot = service.copilot()
                    resp = copilot.generate_campaign(
                        seg=seg_data,
                        totals=tot_data,
                        goal=goal,
                        channel=channel,
                        budget=budget,
                        tone=tone,
                    )

                    engine_tag = "🟢 Gemini 2.5 Flash" if resp.engine == "gemini" else "🟡 Rule-based Copilot (Offline)"
                    st.markdown(f"**Động cơ sinh:** `{engine_tag}`")
                    st.markdown(resp.text)

            st.markdown("---")
            st.subheader(f"📥 Xuất Danh sách Khách hàng Mục tiêu — Nhóm `{sel_seg}`")
            st.caption("Danh sách tích hợp sẵn Mã Voucher, Kênh liên hệ ưu tiên, Ngành hàng gợi ý bán chéo (Next-Best-Category) và Tin nhắn cá nhân hóa cho từng khách hàng.")

            max_export = st.number_input("Số lượng khách hàng xuất tối đa (sắp xếp theo giá trị chi tiêu giảm dần):", min_value=50, max_value=10000, value=500, step=50)
            target_df = service.export_campaign_targets(segment_name=sel_seg, channel=channel, limit=int(max_export))

            st.dataframe(target_df.head(20), width="stretch", hide_index=True)
            csv_bytes = target_df.to_csv(index=False).encode("utf-8-sig")
            safe_seg_slug = sel_seg.lower().replace(" ", "_")
            st.download_button(
                label=f"⬇️ Tải Danh sách {len(target_df):,} Khách hàng Mục tiêu (.CSV)",
                data=csv_bytes,
                file_name=f"campaign_targets_{safe_seg_slug}.csv",
                mime="text/csv",
                type="primary",
            )

        except Exception as e:
            st.error(f"Lỗi: {e}")

    with sub_tab2:
        st.subheader("Hỏi đáp Trực tiếp về Dữ liệu Kinh doanh")
        q = st.text_input("Đặt câu hỏi cho AI (ví dụ: 'Nhóm nào đóng góp nhiều doanh thu nhất và cần ưu tiên gì?'):")
        if st.button("Gửi câu hỏi") and q:
            with st.spinner("Đang tra cứu dữ liệu và phân tích..."):
                profiles = load_segment_profiles()
                tot_data = service.totals()
                copilot = service.copilot()
                resp = copilot.answer_insight(profiles=profiles, totals=tot_data, question=q)
                st.markdown(resp.text)


# ==============================================================================
# TAB 5: QUẢN LÝ MÔ HÌNH, DỮ LIỆU & BÁO CÁO
# ==============================================================================
elif menu.startswith("5."):
    st.markdown('<div class="main-header">⚙️ Quản lý Mô hình, Huấn luyện & Xuất Báo cáo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Quản lý phiên Model Registry, tải lên dữ liệu mới và xuất báo cáo Word hoàn chỉnh</div>', unsafe_allow_html=True)

    tab_report, tab_train, tab_registry = st.tabs(["📄 Xuất Báo cáo Word", "🔄 Huấn luyện Mô hình Mới", "🗄️ Model Registry"])

    with tab_report:
        st.subheader("Xuất Báo cáo Học thuật / Doanh nghiệp Tự động")
        st.write("Tạo tài liệu Word (`.docx`) hoàn chỉnh gồm bảng biểu thống kê, đồ thị phân cụm và khuyến nghị chiến lược.")

        if st.button("📥 Tạo & Tải về Báo cáo Word", type="primary"):
            with st.spinner("Đang kết xuất biểu đồ và biên soạn tài liệu Word..."):
                try:
                    art = get_current_run()
                    out_path = build_report(art, settings.reports_dir)
                    docx_bytes = out_path.read_bytes()
                    st.download_button(
                        label="⬇️ Nhấn vào đây để tải file .docx",
                        data=docx_bytes,
                        file_name=out_path.name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                    st.success(f"Đã tạo báo cáo thành công: `{out_path.name}`")
                except Exception as e:
                    st.error(f"Lỗi tạo báo cáo: {e}")

    with tab_train:
        st.subheader("Huấn luyện Lại Mô hình (Retrain Pipeline)")
        st.caption("Bạn có thể tải lên file CSV giao dịch mới hoặc sử dụng file mặc định cấu hình.")

        up_file = st.file_uploader("Tải file CSV giao dịch mới (tùy chọn):", type=["csv"])
        c_k1, c_k2 = st.columns(2)
        with c_k1:
            custom_k = st.number_input("Ép buộc số cụm k (0 = tự động dò theo Silhouette):", min_value=0, max_value=8, value=0)
        with c_k2:
            run_comp = st.checkbox("Chạy so sánh 6 thuật toán", value=True)

        if st.button("🚀 Khởi chạy Pipeline Huấn luyện", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def on_progress(pct, msg):
                progress_bar.progress(pct)
                status_text.text(f"[{pct:.0%}] {msg}")

            try:
                src = io.BytesIO(up_file.getvalue()) if up_file else settings.data_path
                final_k = custom_k if custom_k > 0 else None
                new_art = pipeline.run_pipeline(
                    source=src,
                    final_k=final_k,
                    compare=run_comp,
                    progress=on_progress,
                )
                service.invalidate_cache()
                st.cache_data.clear()
                st.success(f"🎉 Huấn luyện thành công! Phiên mới: **{new_art.run_id}**")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi huấn luyện: {e}")

    with tab_registry:
        st.subheader("Lịch sử các Phiên Mô hình (Model Registry)")
        reg = service.registry()
        runs = reg.list_runs()
        lat_id = reg.latest_id()

        st.write(f"Phiên đang kích hoạt: **{lat_id}**")
        if runs:
            st.dataframe(pd.DataFrame(runs), width="stretch", hide_index=True)
            sel_run = st.selectbox("Chọn phiên muốn kích hoạt lại:", [r["run_id"] for r in runs])
            if st.button("Kích hoạt phiên này"):
                reg.set_latest(sel_run)
                service.invalidate_cache()
                st.cache_data.clear()
                st.success(f"Đã kích hoạt {sel_run}!")
                st.rerun()
        else:
            st.info("Chưa có phiên mô hình nào được lưu.")
