"""Tiếp nhận dữ liệu: dò encoding, dò tên cột, đọc & chuẩn hóa tên cột, báo cáo chất lượng."""
from __future__ import annotations

import codecs
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import pandas as pd

CANDIDATES = {
    "customer_id": ["customer_id", "customerid", "cust_id", "client_id",
                    "user_id", "userid", "member_id", "id_khach_hang", "makhachhang"],
    "transaction_id": ["transaction_id", "invoice_no", "invoiceno", "invoice_id", "order_id",
                       "receipt_id", "bill_id", "billno", "ma_hoa_don", "basket_id"],
    "date": ["transaction_date", "invoice_date", "order_date", "purchase_date",
             "transaction_datetime", "invoicedate", "date", "ngay_giao_dich"],
    "quantity": ["quantity", "qty", "units", "unit_sold", "quantity_sold", "so_luong"],
    "unit_price": ["unit_price", "unitprice", "price", "selling_price", "don_gia"],
    "amount": ["total_amount", "net_sales", "sales_amount", "total_price", "totalprice",
               "amount", "sales_revenue", "gross_sales", "line_total", "thanh_tien"],
    # Các cột ngữ cảnh tùy chọn phục vụ RFM+ và Gợi ý sản phẩm bán chéo (Next-Best-Product)
    "category": ["product_category", "category", "product_department", "department",
                 "sub_category", "nganh_hang", "loai_san_pham"],
    "product": ["product_name", "item_name", "description", "product_id", "stockcode",
                "sku", "ten_san_pham"],
    "channel": ["sales_channel", "order_channel", "channel", "kenh_ban"],
    "profit": ["gross_profit", "net_profit", "profit", "loi_nhuan"],
    "discount": ["discount_percent", "discount_rate", "discount_amount", "chiet_khau"],
}
ROLES = list(CANDIDATES)
REQUIRED = ["customer_id", "transaction_id", "date"]
LABEL_HINTS = ["segment", "churn", "lifetime_value", "clv", "loyalty_tier",
               "membership_tier", "retention", "risk_category"]


@dataclass
class IngestionResult:
    df: pd.DataFrame
    mapping: dict
    label_columns: list
    report: dict = field(default_factory=dict)


def detect_encoding(sample: bytes) -> str:
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin-1"]:
        try:
            sample.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _read_csv_safe(raw: bytes, preferred_enc: str, **kwargs) -> tuple[pd.DataFrame, str]:
    """Đọc CSV an toàn với cơ chế tự động thử lại qua nhiều bảng mã nếu gặp ký tự đặc biệt (như 0xa3 '£')."""
    candidates = list(dict.fromkeys([preferred_enc, "utf-8-sig", "utf-8", "cp1252", "latin-1"]))
    for enc in candidates:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False, **kwargs), enc
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw), encoding="latin-1", encoding_errors="replace", low_memory=False, **kwargs), "latin-1"


def _clean_id_series(s: pd.Series) -> pd.Series:
    """Chuẩn hóa cột định danh sang chuỗi sạch (bỏ đuôi .0 của float) nhưng GIỮ NGUYÊN NaN để bước clean loại bỏ."""
    mask = s.notna()
    out = pd.Series(pd.NA, index=s.index, dtype="object")
    if not mask.any():
        return out
    vals = s[mask].astype(str).str.strip()
    vals = vals.str.replace(r"^\s*(nan|none|null|<na>)\s*$", "", case=False, regex=True)
    vals = vals.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
    vals = vals.replace("", pd.NA)
    out.loc[mask] = vals
    return out


def detect_column(columns: list[str], role: str, forced: str | None = None) -> str | None:
    """Ưu tiên: cột ép buộc > khớp chính xác (không phân biệt hoa thường) > khớp substring."""
    if forced:
        if forced in columns:
            return forced
        raise ValueError(f"Cột '{forced}' khai báo cho vai trò '{role}' không tồn tại.")
    lower_map = {c.lower(): c for c in columns}
    for cand in CANDIDATES[role]:
        if cand in lower_map:
            return lower_map[cand]
    for c in columns:
        for cand in CANDIDATES[role]:
            if cand in c.lower():
                return c
    return None


def detect_label_columns(columns: list[str]) -> list[str]:
    """Cột nhãn khách hàng có sẵn (bỏ qua thuộc tính sản phẩm product_*)."""
    return [c for c in columns
            if not c.lower().startswith("product") and any(h in c.lower() for h in LABEL_HINTS)]


def resolve_columns(columns: list[str], column_map: dict | None = None) -> dict:
    column_map = column_map or {}
    mapping: dict[str, str | None] = {}
    used: set[str] = set()
    for role in ROLES:
        forced = column_map.get(role)
        avail = columns if forced else [c for c in columns if c not in used]
        col = detect_column(avail, role, forced)
        mapping[role] = col
        if col:
            used.add(col)
    missing = [r for r in REQUIRED if mapping[r] is None]
    if missing:
        raise ValueError(f"Không dò được cột bắt buộc: {missing}. Hãy khai báo thủ công column_map.")
    if not mapping["amount"] and not (mapping["quantity"] and mapping["unit_price"]):
        raise ValueError("Cần cột tổng tiền (amount) HOẶC cả quantity + unit_price để tính Monetary.")
    return mapping


def load_transactions(source: str | Path | BinaryIO | bytes,
                      column_map: dict | None = None) -> IngestionResult:
    """Đọc file CSV giao dịch (đường dẫn, bytes hoặc file-like) và chuẩn hóa tên cột."""
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    elif isinstance(source, bytes):
        raw = source
    else:
        raw = source.read()
    # Quét toàn bộ nội dung byte để phát hiện chính xác ký tự non-UTF8 (như 0xa3 ở cuối file)
    encoding = detect_encoding(raw)

    header_df, encoding = _read_csv_safe(raw, encoding, nrows=0)
    header = header_df.columns.tolist()
    mapping = resolve_columns(header, column_map)
    mapped_cols = {c for c in mapping.values() if c}
    labels = [c for c in detect_label_columns(header) if c not in mapped_cols]
    usecols = list(dict.fromkeys([c for c in mapping.values() if c] + labels))

    df, encoding = _read_csv_safe(raw, encoding, usecols=usecols)
    df = df.rename(columns={col: role.upper() for role, col in mapping.items() if col})
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["CUSTOMER_ID"] = _clean_id_series(df["CUSTOMER_ID"])
    df["TRANSACTION_ID"] = _clean_id_series(df["TRANSACTION_ID"])

    # Bỏ cột nhãn trùng nội dung (vd membership_tier == loyalty_tier)
    dup = [b for i, a in enumerate(labels) for b in labels[i + 1:] if df[a].equals(df[b])]
    dup = list(dict.fromkeys(dup))
    df = df.drop(columns=dup)
    labels = [c for c in labels if c not in dup]

    report = quality_report(df)
    report.update({"encoding": encoding, "duplicate_label_columns": dup})
    return IngestionResult(df=df, mapping=mapping, label_columns=labels, report=report)


def quality_report(df: pd.DataFrame) -> dict:
    shared = (df.groupby("TRANSACTION_ID")["CUSTOMER_ID"].nunique() > 1).mean() * 100
    missing = (df.isna().mean() * 100).round(2)
    return {
        "rows": int(len(df)),
        "customers": int(df["CUSTOMER_ID"].nunique()),
        "transactions": int(df["TRANSACTION_ID"].nunique()),
        "date_min": str(df["DATE"].min().date()) if df["DATE"].notna().any() else None,
        "date_max": str(df["DATE"].max().date()) if df["DATE"].notna().any() else None,
        "shared_transaction_pct": round(float(shared), 2),
        "missing_pct": {k: float(v) for k, v in missing[missing > 0].items()},
    }
