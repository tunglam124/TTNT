"""Hệ thống Gợi ý Bán chéo (Next-Best-Category / Market Basket Association Rules).

Sử dụng lịch sử mua hàng theo khách hàng để tính:
- Luật kết hợp cặp ngành hàng (Support, Confidence P(B|A), Lift) giữa các ngành hàng cùng được mua bởi 1 khách.
- Top ngành hàng & sản phẩm chủ lực theo từng phân khúc RFM.
- Gợi ý Top N ngành hàng/sản phẩm bán chéo cá nhân hóa cho từng khách hàng.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CrossSellRecommender:
    item_col: str | None = None
    rules: list[dict] = field(default_factory=list)
    segment_top_items: dict[str, list[dict]] = field(default_factory=dict)
    global_top_items: list[dict] = field(default_factory=list)
    top_products_by_cat: dict[str, list[str]] = field(default_factory=dict)
    _adj: dict[str, list[dict]] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return bool(self.item_col and self.global_top_items)

    def fit(self, transactions: pd.DataFrame, customers: pd.DataFrame) -> "CrossSellRecommender":
        for cand in ["CATEGORY", "PRODUCT"]:
            if cand in transactions.columns and transactions[cand].notna().any():
                self.item_col = cand
                break
        if not self.item_col:
            return self

        col = self.item_col
        tx = transactions[["CUSTOMER_ID", col, "LINE_TOTAL"]].dropna(subset=[col]).copy()
        tx[col] = tx[col].astype(str)
        n_cust = max(int(tx["CUSTOMER_ID"].nunique()), 1)

        # 1. Độ phổ biến toàn hệ thống
        cat_stats = tx.groupby(col).agg(
            customers=("CUSTOMER_ID", "nunique"),
            revenue=("LINE_TOTAL", "sum"),
        ).sort_values("revenue", ascending=False)
        total_rev = max(float(cat_stats["revenue"].sum()), 1.0)
        cat_stats["support"] = cat_stats["customers"] / n_cust
        cat_stats["revenue_pct"] = cat_stats["revenue"] / total_rev * 100
        self.global_top_items = [
            {"item": str(idx), "customers": int(r["customers"]),
             "support": round(float(r["support"]), 4),
             "revenue": round(float(r["revenue"]), 2),
             "revenue_pct": round(float(r["revenue_pct"]), 2)}
            for idx, r in cat_stats.head(20).iterrows()
        ]

        # 2. Sản phẩm tiêu biểu trong mỗi ngành hàng (nếu có cả CATEGORY và PRODUCT)
        if col == "CATEGORY" and "PRODUCT" in transactions.columns:
            p_df = (transactions.dropna(subset=["CATEGORY", "PRODUCT"])
                    .groupby(["CATEGORY", "PRODUCT"])["LINE_TOTAL"].sum().reset_index()
                    .sort_values("LINE_TOTAL", ascending=False))
            for cat, grp in p_df.groupby("CATEGORY"):
                self.top_products_by_cat[str(cat)] = grp["PRODUCT"].astype(str).head(3).tolist()

        # 3. Top ngành hàng theo từng phân khúc RFM
        if "Segment" in customers.columns:
            merged = tx.merge(customers[["CUSTOMER_ID", "Segment"]], on="CUSTOMER_ID", how="inner")
            for seg, grp in merged.groupby("Segment"):
                seg_rev = max(float(grp["LINE_TOTAL"].sum()), 1.0)
                s_cat = grp.groupby(col).agg(
                    customers=("CUSTOMER_ID", "nunique"),
                    revenue=("LINE_TOTAL", "sum"),
                ).sort_values("revenue", ascending=False)
                self.segment_top_items[str(seg)] = [
                    {"item": str(i), "customers": int(r["customers"]),
                     "revenue_pct": round(float(r["revenue"] / seg_rev * 100), 1),
                     "sample_products": self.top_products_by_cat.get(str(i), [])}
                    for i, r in s_cat.head(5).iterrows()
                ]

        # 4. Ma trận đồng xuất hiện (Market Basket Co-occurrence) cấp khách hàng
        pairs = tx[["CUSTOMER_ID", col]].drop_duplicates()
        top_items = set(cat_stats.head(50).index.astype(str))
        pairs = pairs[pairs[col].isin(top_items)]

        items_list = sorted(top_items)
        if len(items_list) >= 2:
            cat_to_idx = {c: i for i, c in enumerate(items_list)}
            cust_codes = pairs["CUSTOMER_ID"].astype("category").cat.codes.to_numpy()
            item_codes = pairs[col].map(cat_to_idx).to_numpy()
            from scipy.sparse import csr_matrix
            mat = csr_matrix(
                (np.ones(len(pairs), dtype=np.float32), (cust_codes, item_codes)),
                shape=(int(cust_codes.max()) + 1, len(items_list)),
            )
            cooc = (mat.T @ mat).toarray()
            diag = np.diag(cooc).clip(min=1.0)
            support_vec = diag / n_cust

            rules: list[dict] = []
            adj: dict[str, list[dict]] = {c: [] for c in items_list}
            min_both = 5 if n_cust >= 50 else 1
            for i, a in enumerate(items_list):
                for j, b in enumerate(items_list):
                    if i == j:
                        continue
                    both = float(cooc[i, j])
                    if both < min_both:
                        continue
                    conf = both / float(diag[i])
                    lift = conf / float(support_vec[j]) if support_vec[j] > 0 else 0.0
                    rule = {
                        "antecedent": a,
                        "consequent": b,
                        "pair_customers": int(both),
                        "support": round(both / n_cust, 4),
                        "confidence": round(conf, 4),
                        "lift": round(lift, 3),
                        "top_products": self.top_products_by_cat.get(b, []),
                    }
                    rules.append(rule)
                    adj[a].append(rule)

            rules.sort(key=lambda r: (r["lift"], r["confidence"], r["pair_customers"]), reverse=True)
            for a in adj:
                adj[a].sort(key=lambda r: (r["lift"] * r["confidence"], r["pair_customers"]), reverse=True)
            self.rules = rules[:30]
            self._adj = adj

        return self

    def recommend_for_customer(self, bought_items: list[str], segment: str | None = None,
                               top_n: int = 3) -> list[dict]:
        """Gợi ý Top N ngành hàng khách CHƯA mua (hoặc mua kèm có Lift cao nhất)."""
        if not self.available:
            return []
        bought_set = {str(x) for x in bought_items if pd.notna(x)}
        scores: dict[str, dict] = {}

        for a in bought_set:
            for r in self._adj.get(a, []):
                b = r["consequent"]
                if b in bought_set:
                    continue
                sc = r["lift"] * r["confidence"]
                if b not in scores or sc > scores[b]["score"]:
                    scores[b] = {
                        "category": b,
                        "recommended_category": b,
                        "score": sc,
                        "confidence": r["confidence"],
                        "lift": r["lift"],
                        "based_on": a,
                        "reason": f"Thường mua kèm '{a}' (Confidence {r['confidence']:.1%}, Lift {r['lift']:.2f}x)",
                        "sample_products": self.top_products_by_cat.get(b, []),
                    }

        ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)[:top_n]

        # Bổ sung từ top ngành hàng của phân khúc nếu khách chưa đủ top_n gợi ý
        fallback_pool = self.segment_top_items.get(str(segment), []) + self.global_top_items
        chosen = {r["category"] for r in ranked}
        for item_info in fallback_pool:
            if len(ranked) >= top_n:
                break
            b = item_info["item"]
            if b in bought_set or b in chosen:
                continue
            chosen.add(b)
            ranked.append({
                "category": b,
                "recommended_category": b,
                "score": 0.0,
                "confidence": 0.0,
                "lift": 1.0,
                "based_on": segment or "Toàn hệ thống",
                "reason": f"Ngành hàng chủ lực của nhóm {segment or 'khách hàng'} ({item_info['revenue_pct']:.1f}% doanh thu)",
                "sample_products": self.top_products_by_cat.get(b, []),
            })
        return ranked
