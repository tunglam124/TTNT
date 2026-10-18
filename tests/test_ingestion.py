"""Kiểm thử tự động cho module ingestion và dò tên cột."""
import io
import pandas as pd
import pytest

from src.data.ingestion import detect_column, detect_encoding, resolve_columns, load_transactions


def test_detect_encoding():
    utf8_bytes = "Mã khách hàng,Ngày mua".encode("utf-8")
    assert detect_encoding(utf8_bytes) in ["utf-8", "utf-8-sig"]


def test_detect_column():
    cols = ["MaKhachHang", "Basket_ID", "Invoice_Date", "Line_Total"]
    assert detect_column(cols, "customer_id") == "MaKhachHang"
    assert detect_column(cols, "transaction_id") == "Basket_ID"
    assert detect_column(cols, "date") == "Invoice_Date"
    assert detect_column(cols, "amount") == "Line_Total"


def test_resolve_columns_missing_required():
    with pytest.raises(ValueError):
        resolve_columns(["customer_id", "date"])  # thiếu transaction_id


def test_load_transactions_from_csv():
    csv_text = """customer_id,transaction_id,date,amount
C01,T01,2024-01-01,100.0
C01,T02,2024-01-10,200.0
C02,T03,2024-02-01,150.0
"""
    res = load_transactions(io.BytesIO(csv_text.encode("utf-8")))
    assert len(res.df) == 3
    assert "CUSTOMER_ID" in res.df.columns
    assert "TRANSACTION_ID" in res.df.columns
    assert "DATE" in res.df.columns
    assert "AMOUNT" in res.df.columns


def test_load_transactions_cp1252_byte_0xa3_and_nan_customer():
    from src.data.rfm_engine import clean_transactions

    # Mô phỏng file CSV có đoạn đầu toàn ASCII nhưng phía sau chứa ký tự '£' (0xa3 trong cp1252)
    # kèm theo CustomerID dạng số thực 17850.0 và dòng thiếu CustomerID
    header = "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID\n"
    ascii_rows = "536365,85123A,WHITE HANGING HEART,6,2010-12-01 08:26:00,2.55,17850.0\n" * 1500
    pound_row = "536366,22633,HAND WARMER £2.10,4,2010-12-01 09:00:00,2.10,13047.0\n"
    nan_row = "536367,22634,UNKNOWN CUSTOMER ITEM,2,2010-12-01 10:00:00,1.50,\n"
    raw_bytes = (header + ascii_rows + pound_row + nan_row).encode("cp1252")
    assert b"\xa3" in raw_bytes

    ing = load_transactions(io.BytesIO(raw_bytes))
    assert ing.report["encoding"] in ("cp1252", "latin-1")
    cleaned = clean_transactions(ing.df)
    cids = set(cleaned["CUSTOMER_ID"].unique())
    assert cids == {"17850", "13047"}

