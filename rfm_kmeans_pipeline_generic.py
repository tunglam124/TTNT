"""
====================================================================
PIPELINE TỔNG QUÁT: PHÂN CỤM KHÁCH HÀNG BẰNG RFM + K-MEANS
Chạy được trên NHIỀU bộ dữ liệu giao dịch bán lẻ khác nhau
(Online Retail - UCI, Supermarket - Kaggle, hoặc dataset khác),
miễn là dữ liệu có các trường: khách hàng, giao dịch, ngày, và
số tiền/số lượng+đơn giá.

CÁCH HOẠT ĐỘNG:
- Bạn CHỈ cần sửa phần CONFIG bên dưới khi đổi sang bộ dữ liệu khác.
- Nếu để COLUMN_MAP = None cho một vai trò nào đó, chương trình sẽ
  TỰ ĐỘNG dò tên cột phù hợp dựa trên danh sách từ khóa thường gặp
  (CANDIDATES). Nếu dò sai hoặc không tìm thấy, bạn tự điền tên cột
  chính xác vào COLUMN_MAP để ép buộc dùng đúng cột đó.
- Toàn bộ 7 giai đoạn xử lý bên dưới đều thao tác trên các cột đã
  được CHUẨN HÓA TÊN (CUSTOMER_ID, TXN_ID, DATE, AMOUNT...), nên
  logic xử lý không cần sửa khi đổi dataset.

Yêu cầu thư viện:
    pip install pandas numpy scikit-learn matplotlib seaborn --break-system-packages
"""

import os
import codecs
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (kích hoạt chế độ vẽ 3D)

warnings.filterwarnings("ignore")

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


def fmt_num(x, decimals=0):
    """Định dạng số có dấu phẩy ngăn cách hàng nghìn, dễ đọc hơn số thô."""
    if pd.isna(x):
        return "—"
    return f"{x:,.{decimals}f}"


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def compute_cluster_metrics(X, labels):
    """Tính các chỉ số đánh giá chất lượng phân cụm (dùng chung cho stage4b và stage5)."""
    mask = labels != -1
    X_valid, labels_valid = X[mask], labels[mask]
    if len(set(labels_valid)) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    return {
        "silhouette": silhouette_score(X_valid, labels_valid),
        "davies_bouldin": davies_bouldin_score(X_valid, labels_valid),
        "calinski_harabasz": calinski_harabasz_score(X_valid, labels_valid),
    }



# ====================================================================
# 0. CONFIG — PHẦN DUY NHẤT CẦN SỬA KHI ĐỔI BỘ DỮ LIỆU
# ====================================================================
DATA_PATH = r"D:\archive (6)\supermarket_large_dataset.csv"     # <-- Đường dẫn tới file CSV của bạn
OUTPUT_DIR = "outputs_rfm"
RANDOM_STATE = 42
FINAL_K = None                    # None = tự động dùng best_k từ Giai đoạn 4 (Silhouette). Đặt số cụ thể (vd: 4) để ép buộc.
RUN_ALGORITHM_COMPARISON = True   # True: chạy Giai đoạn 4.5 so sánh 6 thuật toán phân cụm (giống bài báo JABES)
MAX_SAMPLE_FOR_HEAVY_ALGOS = 3000  # Spectral Clustering, Mean Shift và Dunn Index chạy chậm với dữ liệu lớn -> lấy mẫu tối đa chừng này khách hàng

# Khai báo TÊN CỘT THẬT trong file của bạn. Để None để tự động dò.
# Ví dụ ép buộc thủ công: "customer_id": "MaKhachHang"
COLUMN_MAP = {
    "customer_id":    None,   # Mã khách hàng (bắt buộc)
    "transaction_id": None,   # Mã đơn hàng/hóa đơn/giỏ hàng, dùng để đếm Frequency (bắt buộc)
    "date":           None,   # Ngày giao dịch (bắt buộc)
    "quantity":       None,   # Số lượng (tùy chọn — cần nếu không có sẵn cột amount)
    "unit_price":     None,   # Đơn giá (tùy chọn — cần nếu không có sẵn cột amount)
    "amount":         None,   # Tổng tiền dòng giao dịch, nếu dataset đã tính sẵn (tùy chọn, ưu tiên dùng nếu có)
}

# Các cột "nhãn có sẵn" (ground truth) muốn đối chiếu ở Giai đoạn 6, nếu có trong data.
# Để trống [] nếu dataset không có, hoặc để None để tự động dò các cột có tên gợi ý phân khúc/churn/CLV.
EXTRA_LABEL_COLUMNS = None


# ====================================================================
# 1. TỪ ĐIỂN TỪ KHÓA DÙNG ĐỂ TỰ ĐỘNG DÒ TÊN CỘT
# ====================================================================
CANDIDATES = {
    "customer_id":    ["customer_id", "customerid", "cust_id", "client_id",
                        "user_id", "userid", "member_id", "id_khach_hang", "makhachhang"],
    "transaction_id": ["basket_id", "invoice_no", "invoiceno", "invoice_id", "order_id",
                        "transaction_id", "receipt_id", "bill_id", "billno", "ma_hoa_don"],
    "date":           ["transaction_date", "invoice_date", "order_date", "purchase_date",
                        "transaction_datetime", "invoicedate", "date", "ngay_giao_dich"],
    "quantity":       ["quantity", "qty", "units", "unit_sold", "quantity_sold", "so_luong"],
    "unit_price":     ["unit_price", "unitprice", "price", "selling_price", "don_gia"],
    "amount":         ["total_amount", "net_sales", "sales_amount", "total_price", "totalprice",
                        "amount", "sales_revenue", "gross_sales", "line_total", "thanh_tien"],
}

LABEL_HINTS = ["segment", "churn", "lifetime_value", "clv", "loyalty_tier",
               "membership_tier", "retention", "risk_category"]


def detect_encoding(path, _cache={}):
    """Tự dò encoding của file CSV (utf-8, utf-8-sig, cp1252, cuối cùng latin-1).
    Chỉ đọc 2 MB đầu để tiết kiệm I/O; cache kết quả theo đường dẫn."""
    if path in _cache:
        return _cache[path]
    for enc in ["utf-8", "utf-8-sig", "cp1252"]:
        dec = codecs.getincrementaldecoder(enc)()
        try:
            with open(path, "rb") as f:
                sample = f.read(2 << 20)  # 2 MB đủ để nhận dạng encoding
                dec.decode(sample, final=(len(sample) < 2 << 20))
            _cache[path] = enc
            return enc
        except UnicodeDecodeError:
            continue
    _cache[path] = "latin-1"
    return "latin-1"


def detect_column(columns, role, forced=None):
    """Tìm tên cột thật khớp với một 'vai trò' logic (customer_id, date...).
    Ưu tiên: tên do người dùng ép buộc > khớp chính xác (không phân biệt hoa/thường)
    > khớp một phần (substring)."""
    if forced:
        if forced in columns:
            return forced
        raise ValueError(f"Cột '{forced}' bạn khai báo cho vai trò '{role}' không tồn tại trong file dữ liệu.")

    lower_map = {c.lower(): c for c in columns}
    for cand in CANDIDATES.get(role, []):
        if cand in lower_map:
            return lower_map[cand]
    for c in columns:
        for cand in CANDIDATES.get(role, []):
            if cand in c.lower():
                return c
    return None


def detect_label_columns(columns, forced_list=None):
    """Tự động tìm các cột nhãn có sẵn (segment/churn/CLV...) để đối chiếu ở Giai đoạn 6."""
    if forced_list is not None:
        return [c for c in forced_list if c in columns]
    found = []
    for c in columns:
        if any(hint in c.lower() for hint in LABEL_HINTS):
            found.append(c)
    return found


# ====================================================================
# GIAI ĐOẠN 0: DÒ CỘT & BÁO CÁO MAPPING (bước mới — cốt lõi của bản tổng quát)
# ====================================================================
def stage0_resolve_columns(path: str, column_map: dict, extra_label_columns):
    print_section("GIAI ĐOẠN 0: DÒ TÊN CỘT TỰ ĐỘNG")

    header = pd.read_csv(path, nrows=0, encoding=detect_encoding(path)).columns.tolist()

    resolved = {}
    for role in ["customer_id", "transaction_id", "date", "quantity", "unit_price", "amount"]:
        col = detect_column(header, role, forced=column_map.get(role))
        resolved[role] = col
        trang_thai = "OK" if col else ("BẮT BUỘC nhưng KHÔNG TÌM THẤY" if role in
                                        ("customer_id", "transaction_id", "date") else "không có (tùy chọn)")
        print(f"  {role:15s} -> {str(col):25s} [{trang_thai}]")

    # Kiểm tra các cột bắt buộc
    required = ["customer_id", "transaction_id", "date"]
    missing_required = [r for r in required if resolved[r] is None]
    if missing_required:
        raise ValueError(
            f"Không tự dò được các cột bắt buộc: {missing_required}.\n"
            f"=> Hãy mở file dữ liệu, xem tên cột thật, rồi điền thủ công vào COLUMN_MAP ở đầu script."
        )

    # Kiểm tra đủ điều kiện tính Monetary: cần 'amount' HOẶC cả 'quantity' + 'unit_price'
    if not resolved["amount"] and not (resolved["quantity"] and resolved["unit_price"]):
        raise ValueError(
            "Không đủ dữ liệu để tính Monetary: cần có cột tổng tiền (amount) "
            "HOẶC cả hai cột quantity + unit_price. Hãy điền thủ công vào COLUMN_MAP."
        )

    labels = detect_label_columns(header, extra_label_columns)
    print(f"\n  Cột nhãn có sẵn để đối chiếu (Giai đoạn 6): {labels if labels else 'Không tìm thấy / không dùng'}")

    return resolved, labels


# ====================================================================
# GIAI ĐOẠN 1: THU THẬP & MÔ TẢ DỮ LIỆU (Data Understanding)
# ====================================================================
def stage1_load_and_explore(path: str, resolved: dict, labels: list) -> pd.DataFrame:
    print_section("GIAI ĐOẠN 1: THU THẬP & MÔ TẢ DỮ LIỆU")

    usecols = [c for c in resolved.values() if c] + labels
    usecols = list(dict.fromkeys(usecols))  # loại trùng, giữ thứ tự

    df = pd.read_csv(path, usecols=usecols, encoding=detect_encoding(path))

    # Đổi tên cột về dạng chuẩn nội bộ, để các giai đoạn sau không phụ thuộc dataset gốc
    rename_map = {resolved[role]: role.upper() for role in
                  ["customer_id", "transaction_id", "date", "quantity", "unit_price", "amount"]
                  if resolved[role]}
    df = df.rename(columns=rename_map)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    print(f"  {'Số dòng dữ liệu:':32s} {fmt_num(len(df))}")
    print(f"  {'Số cột:':32s} {df.shape[1]}")
    print(f"  {'Khoảng thời gian:':32s} {df['DATE'].min().date()}  ->  {df['DATE'].max().date()}")
    print(f"  {'Số khách hàng duy nhất:':32s} {fmt_num(df['CUSTOMER_ID'].nunique())}")
    print(f"  {'Số giao dịch duy nhất:':32s} {fmt_num(df['TRANSACTION_ID'].nunique())}")

    missing = (df.isna().mean() * 100).round(2)
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print("\n  Cột có dữ liệu thiếu:")
        for col, pct in missing.items():
            print(f"    - {col:20s}: {pct:5.2f} %")
    else:
        print("\n  Không có cột nào thiếu dữ liệu.")

    return df


# ====================================================================
# GIAI ĐOẠN 2: LÀM SẠCH & TIỀN XỬ LÝ (Data Preparation)
# ====================================================================
def stage2_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print_section("GIAI ĐOẠN 2: LÀM SẠCH & TIỀN XỬ LÝ DỮ LIỆU")

    n_before = len(df)

    df = df.dropna(subset=["CUSTOMER_ID", "TRANSACTION_ID", "DATE"])

    if "QUANTITY" in df.columns:
        df = df[df["QUANTITY"] > 0]
    if "UNIT_PRICE" in df.columns:
        df = df[df["UNIT_PRICE"] > 0]

    df = df.drop_duplicates()

    # Tính LINE_TOTAL: ưu tiên cột AMOUNT có sẵn, nếu không thì quantity * unit_price
    if "AMOUNT" in df.columns and df["AMOUNT"].notna().any():
        df["LINE_TOTAL"] = df["AMOUNT"]
    else:
        df["LINE_TOTAL"] = df["QUANTITY"] * df["UNIT_PRICE"]

    df = df[df["LINE_TOTAL"] > 0]

    # Capping IQR để xử lý ngoại lai
    Q1, Q3 = df["LINE_TOTAL"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    n_outliers = (df["LINE_TOTAL"] > upper_bound).sum()
    df["LINE_TOTAL"] = df["LINE_TOTAL"].clip(upper=upper_bound)

    pct_loai = (n_before - len(df)) / n_before * 100
    print(f"  {'Số dòng trước xử lý:':38s} {fmt_num(n_before)}")
    print(f"  {'Số dòng sau xử lý:':38s} {fmt_num(len(df))}  (đã loại {fmt_num(n_before - len(df))} dòng, {pct_loai:.1f}%)")
    print(f"  {'Ngưỡng trên xử lý ngoại lai (IQR):':38s} {fmt_num(upper_bound, 0)}")
    print(f"  {'Số giá trị ngoại lai bị giới hạn:':38s} {fmt_num(n_outliers)} ({n_outliers/len(df)*100:.1f}% dữ liệu sau làm sạch)")

    return df


# ====================================================================
# GIAI ĐOẠN 3: TÍNH TOÁN CHỈ SỐ RFM (Feature Engineering)
# ====================================================================
def stage3_compute_rfm(df: pd.DataFrame, labels: list) -> pd.DataFrame:
    print_section("GIAI ĐOẠN 3: TÍNH TOÁN CHỈ SỐ RFM")

    snapshot_date = df["DATE"].max() + pd.Timedelta(days=1)
    print(f"Ngày mốc tham chiếu (snapshot date): {snapshot_date.date()}")

    rfm = df.groupby("CUSTOMER_ID").agg(
        Recency=("DATE", lambda x: (snapshot_date - x.max()).days),
        Frequency=("TRANSACTION_ID", "nunique"),
        Monetary=("LINE_TOTAL", "sum"),
    ).reset_index()

    if labels:
        extra = df.groupby("CUSTOMER_ID")[labels].first().reset_index()
        rfm = rfm.merge(extra, on="CUSTOMER_ID", how="left")

    print(f"  {'Số khách hàng sau khi tính RFM:':32s} {fmt_num(len(rfm))}")

    desc = rfm[["Recency", "Frequency", "Monetary"]].describe().T
    print("\n  Thống kê mô tả 3 chỉ số RFM:")
    header = f"  {'Chỉ số':<12}{'Nhỏ nhất':>14}{'Trung vị':>14}{'Trung bình':>14}{'Lớn nhất':>16}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for idx in ["Recency", "Frequency", "Monetary"]:
        r = desc.loc[idx]
        dv = 2 if idx == "Monetary" else 0
        print(f"  {idx:<12}{fmt_num(r['min'], dv):>14}{fmt_num(r['50%'], dv):>14}{fmt_num(r['mean'], 1):>14}{fmt_num(r['max'], dv):>16}")

    rfm.to_csv(f"{OUTPUT_DIR}/rfm_table.csv", index=False)
    print(f"\n  Đã lưu bảng RFM vào: {OUTPUT_DIR}/rfm_table.csv")

    return rfm


# ====================================================================
# GIAI ĐOẠN 4: CHUẨN HÓA & XÁC ĐỊNH SỐ CỤM TỐI ƯU
# ====================================================================
def stage4_scale_and_find_k(rfm: pd.DataFrame, k_range=range(2, 9)):
    print_section("GIAI ĐOẠN 4: CHUẨN HÓA DỮ LIỆU & XÁC ĐỊNH SỐ CỤM TỐI ƯU")

    rfm_log = rfm[["Recency", "Frequency", "Monetary"]].apply(lambda x: np.log1p(x))
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    wcss, silhouette_scores = [], []
    best_sil = -1.0
    print(f"  {'k':>3}{'WCSS':>14}{'Silhouette':>14}")
    print("  " + "-" * 31)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        lb = km.fit_predict(rfm_scaled)
        wcss.append(km.inertia_)
        sil = silhouette_score(rfm_scaled, lb)
        silhouette_scores.append(sil)
        if sil > best_sil:
            best_sil = sil
        danh_dau = "  <- cao nhất" if sil >= best_sil else ""
        print(f"  {k:>3}{fmt_num(km.inertia_, 1):>14}{sil:>14.3f}{danh_dau}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), wcss, marker="o")
    axes[0].set_title("Phương pháp Elbow (WCSS)")
    axes[0].set_xlabel("Số cụm (k)")
    axes[1].plot(list(k_range), silhouette_scores, marker="o", color="orange")
    axes[1].set_title("Chỉ số Silhouette theo k")
    axes[1].set_xlabel("Số cụm (k)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/elbow_silhouette.png", dpi=150)
    plt.close()

    best_k = list(k_range)[int(np.argmax(silhouette_scores))]
    print(f"\n  >>> Số cụm đề xuất theo Silhouette cao nhất: k = {best_k}")
    print(f"  >>> Lưu ý: nên đối chiếu thêm với biểu đồ Elbow và khả năng diễn giải kinh doanh trước khi chốt k.")

    return rfm_scaled, best_k


# ====================================================================
# GIAI ĐOẠN 4.5: SO SÁNH NHIỀU THUẬT TOÁN PHÂN CỤM (giống Bảng 3, Bảng 4, Hình 3 của JABES 2025)
# ====================================================================
def _dunn_index(X, labels, max_sample=MAX_SAMPLE_FOR_HEAVY_ALGOS, random_state=RANDOM_STATE):
    """Dunn Index = khoảng cách nhỏ nhất giữa 2 cụm khác nhau / khoảng cách lớn nhất trong nội bộ 1 cụm.
    Càng CAO càng tốt. Tính trên toàn bộ dữ liệu tốn O(n^2) bộ nhớ nên với dữ liệu lớn sẽ lấy mẫu."""
    X, labels = np.asarray(X), np.asarray(labels)
    if len(X) > max_sample:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), max_sample, replace=False)
        X, labels = X[idx], labels[idx]

    valid = labels != -1  # bỏ điểm nhiễu của DBSCAN (nhãn -1)
    X, labels = X[valid], labels[valid]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.nan

    # Tính từng cặp cụm riêng thay vì cấp phát ma trận N×N đầy đủ
    cluster_indices = {lb: np.where(labels == lb)[0] for lb in unique_labels}

    max_intra = 0.0
    for lb, idx in cluster_indices.items():
        if len(idx) > 1:
            D_intra = pairwise_distances(X[idx])
            max_intra = max(max_intra, D_intra.max())

    min_inter = np.inf
    labels_list = list(unique_labels)
    for i, l1 in enumerate(labels_list):
        for l2 in labels_list[i + 1:]:
            D_inter = pairwise_distances(X[cluster_indices[l1]], X[cluster_indices[l2]])
            min_inter = min(min_inter, D_inter.min())

    return min_inter / max_intra if max_intra > 0 else np.nan


def _fuzzy_c_means(X, c, m=2.0, max_iter=150, tol=1e-5, random_state=RANDOM_STATE):
    """Cài đặt gọn của thuật toán Fuzzy C-Means (không cần thư viện ngoài).
    Trả về nhãn cụm cứng (mỗi điểm gán vào cụm có độ thuộc cao nhất)."""
    if m <= 1.0:
        raise ValueError("Tham số fuzziness m phải > 1.0 (thường dùng m=2.0)")
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    u = rng.rand(n, c)
    u /= u.sum(axis=1, keepdims=True)

    for _ in range(max_iter):
        um = u ** m
        centers = (um.T @ X) / um.sum(axis=0)[:, None]
        dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        dist = np.fmax(dist, 1e-10)
        new_u = 1.0 / (dist ** (2.0 / (m - 1)))
        new_u /= new_u.sum(axis=1, keepdims=True)
        if np.linalg.norm(new_u - u) < tol:
            u = new_u
            break
        u = new_u

    return np.argmax(u, axis=1)


def stage4b_compare_algorithms(rfm_scaled: np.ndarray, k: int):
    """So sánh 6 thuật toán phân cụm bằng 4 chỉ số định lượng, tương tự Bảng 3 trong
    Nguyễn Tịnh Giang & Hoàng Anh (2025). K-Means, GMM, Fuzzy C-Means và Spectral Clustering
    được ép chạy với đúng k cụm; Mean Shift và DBSCAN tự xác định số cụm theo mật độ dữ liệu."""
    print_section("GIAI ĐOẠN 4.5: SO SÁNH NHIỀU THUẬT TOÁN PHÂN CỤM")

    n = len(rfm_scaled)
    heavy_note = ""
    X_heavy = rfm_scaled
    if n > MAX_SAMPLE_FOR_HEAVY_ALGOS:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(n, MAX_SAMPLE_FOR_HEAVY_ALGOS, replace=False)
        X_heavy = rfm_scaled[idx]
        heavy_note = f" (lấy mẫu {MAX_SAMPLE_FOR_HEAVY_ALGOS:,}/{n:,} khách hàng vì thuật toán chạy chậm với dữ liệu lớn)"

    algorithms = {}

    print("  Đang chạy K-Means...")
    _kmeans_obj = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    _kmeans_labels = _kmeans_obj.fit_predict(rfm_scaled)
    algorithms["K-Means"] = (_kmeans_labels, rfm_scaled)

    print(f"  Đang chạy Spectral Clustering...{heavy_note}")
    try:
        sc_labels = SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=10,
                                        assign_labels="kmeans", random_state=RANDOM_STATE).fit_predict(X_heavy)
        algorithms["Spectral Clustering"] = (sc_labels, X_heavy)
    except Exception as e:
        print(f"    [Bỏ qua] Spectral Clustering lỗi: {e}")

    print(f"  Đang chạy Mean Shift...{heavy_note}")
    try:
        bandwidth = estimate_bandwidth(X_heavy, quantile=0.2, random_state=RANDOM_STATE)
        if bandwidth <= 0:
            bandwidth = None
        ms_labels = MeanShift(bandwidth=bandwidth).fit_predict(X_heavy)
        algorithms["Mean Shift"] = (ms_labels, X_heavy)
    except Exception as e:
        print(f"    [Bỏ qua] Mean Shift lỗi: {e}")

    print("  Đang chạy Gaussian Mixture Model (GMM)...")
    algorithms["GMM"] = (GaussianMixture(n_components=k, random_state=RANDOM_STATE).fit_predict(rfm_scaled), rfm_scaled)

    print("  Đang chạy DBSCAN...")
    algorithms["DBSCAN"] = (DBSCAN(eps=0.5, min_samples=10).fit_predict(rfm_scaled), rfm_scaled)

    print("  Đang chạy Fuzzy C-Means (FCM)...")
    algorithms["FCM"] = (_fuzzy_c_means(rfm_scaled, c=k, random_state=RANDOM_STATE), rfm_scaled)

    rows = []
    label_store = {}
    for name, val in algorithms.items():
        labels, X_used = val
        label_store[name] = (labels, X_used)

        n_clusters = len(set(labels) - {-1})
        if n_clusters < 2:
            rows.append({"Thuật toán": name, "Số cụm": n_clusters, "Silhouette": np.nan,
                         "Davies-Bouldin": np.nan, "Calinski-Harabasz": np.nan, "Dunn": np.nan,
                         "Nhận xét": "Không đủ cụm để tính chỉ số (< 2 cụm)"})
            continue

        mask = labels != -1
        m = compute_cluster_metrics(X_used, labels)
        dunn = _dunn_index(X_used[mask], labels[mask])

        rows.append({"Thuật toán": name, "Số cụm": n_clusters, "Silhouette": round(m["silhouette"], 3),
                     "Davies-Bouldin": round(m["davies_bouldin"], 3), "Calinski-Harabasz": round(m["calinski_harabasz"], 1),
                     "Dunn": round(dunn, 3) if not np.isnan(dunn) else np.nan, "Nhận xét": ""})

    result = pd.DataFrame(rows).set_index("Thuật toán")

    # Xếp hạng để gợi ý thuật toán tốt nhất (chỉ xét các dòng đủ điều kiện)
    valid_rows = result.dropna(subset=["Silhouette"])
    if len(valid_rows) > 0:
        rank = (valid_rows["Silhouette"].rank(ascending=False)
                + valid_rows["Davies-Bouldin"].rank(ascending=True)
                + valid_rows["Calinski-Harabasz"].rank(ascending=False))
        best_algo = rank.idxmin()
        result.loc[best_algo, "Nhận xét"] = "<< Đề xuất: cân bằng tốt nhất giữa các chỉ số"

    print("\n  Bảng so sánh chất lượng phân cụm (tương tự Bảng 3, JABES 2025):")
    header = f"  {'Thuật toán':<20}{'Số cụm':>8}{'Silhouette':>12}{'Davies-B.':>12}{'Calinski-H.':>14}{'Dunn':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for algo, row in result.iterrows():
        sil = f"{row['Silhouette']:.3f}" if pd.notna(row["Silhouette"]) else "—"
        db = f"{row['Davies-Bouldin']:.3f}" if pd.notna(row["Davies-Bouldin"]) else "—"
        ch = fmt_num(row["Calinski-Harabasz"], 1) if pd.notna(row["Calinski-Harabasz"]) else "—"
        dunn = f"{row['Dunn']:.3f}" if pd.notna(row["Dunn"]) else "—"
        print(f"  {algo:<20}{row['Số cụm']:>8}{sil:>12}{db:>12}{ch:>14}{dunn:>9}   {row['Nhận xét']}")

    result.to_csv(f"{OUTPUT_DIR}/algorithm_comparison.csv")
    print(f"\n  Đã lưu bảng so sánh vào: {OUTPUT_DIR}/algorithm_comparison.csv")

    _visualize_algorithm_metrics(result)
    _visualize_algorithm_comparison(label_store)

    return result, label_store, _kmeans_labels


def _visualize_algorithm_metrics(result: pd.DataFrame):
    """Vẽ 4 biểu đồ cột so sánh Silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn
    giữa các thuật toán — giúp nhìn nhanh thuật toán nào vượt trội mà không cần đọc bảng số."""
    metrics = [
        ("Silhouette", "Càng CAO càng tốt", "#2E7D32"),
        ("Davies-Bouldin", "Càng THẤP càng tốt", "#C62828"),
        ("Calinski-Harabasz", "Càng CAO càng tốt", "#1565C0"),
        ("Dunn", "Càng CAO càng tốt", "#EF6C00"),
    ]
    plot_df = result.dropna(subset=["Silhouette"])
    if plot_df.empty:
        print("  [Bỏ qua vẽ biểu đồ so sánh chỉ số] Không có thuật toán nào đủ điều kiện tính chỉ số.")
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, (col, ghi_chu, color) in zip(axes, metrics):
        vals = plot_df[col]
        bars = ax.bar(vals.index, vals.values, color=color, alpha=0.85)
        best_idx = vals.idxmax() if "CAO" in ghi_chu else vals.idxmin()
        bars[list(vals.index).index(best_idx)].set_edgecolor("black")
        bars[list(vals.index).index(best_idx)].set_linewidth(2.5)
        ax.set_title(f"{col}\n({ghi_chu})", fontsize=10.5, fontweight="bold")
        ax.tick_params(axis="x", rotation=35)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals.values):
            ax.text(i, v, f"{v:,.2f}" if col != "Calinski-Harabasz" else f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=8)

    plt.suptitle("So sánh 4 chỉ số đánh giá chất lượng phân cụm giữa các thuật toán "
                  "(viền đen = thuật toán tốt nhất theo từng chỉ số)", fontsize=11, y=1.04)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/algorithm_comparison_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Đã lưu biểu đồ so sánh chỉ số vào: {OUTPUT_DIR}/algorithm_comparison_metrics.png")


def _visualize_algorithm_comparison(label_store: dict):
    """Vẽ lưới scatter 3D (Recency-Frequency-Monetary đã chuẩn hóa) cho từng thuật toán,
    tương tự Hình 3 trong bài báo tham chiếu."""
    n_algo = len(label_store)
    n_cols = 3
    n_rows = int(np.ceil(n_algo / n_cols))

    fig = plt.figure(figsize=(5.5 * n_cols, 4.8 * n_rows))
    for i, (name, (labels, X_used)) in enumerate(label_store.items(), start=1):
        ax = fig.add_subplot(n_rows, n_cols, i, projection="3d")
        sc = ax.scatter(X_used[:, 0], X_used[:, 1], X_used[:, 2], c=labels, cmap="tab10", s=12, alpha=0.7)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Recency (chuẩn hóa)", fontsize=8)
        ax.set_ylabel("Frequency (chuẩn hóa)", fontsize=8)
        ax.set_zlabel("Monetary (chuẩn hóa)", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/algorithm_comparison_3d.png", dpi=150)
    plt.close()
    print(f"Đã lưu biểu đồ so sánh 3D vào: {OUTPUT_DIR}/algorithm_comparison_3d.png")


# ====================================================================
# GIAI ĐOẠN 5: HUẤN LUYỆN K-MEANS & GÁN NHÃN CỤM
# ====================================================================
def stage5_train_kmeans(rfm: pd.DataFrame, rfm_scaled: np.ndarray, k: int,
                       precomputed_labels: np.ndarray = None) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(f"GIAI ĐOẠN 5: HUẤN LUYỆN K-MEANS VỚI k={k} & GÁN NHÃN CỤM")
    print("=" * 70)

    rfm = rfm.copy()
    if precomputed_labels is not None:
        rfm["Cluster"] = precomputed_labels
    else:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    m = compute_cluster_metrics(rfm_scaled, rfm["Cluster"].values)
    print(f"  {'Silhouette Score:':28s} {m['silhouette']:.3f}   (càng cao càng tốt, tối đa 1)")
    print(f"  {'Davies-Bouldin Index:':28s} {m['davies_bouldin']:.3f}   (càng thấp càng tốt)")
    print(f"  {'Calinski-Harabasz Index:':28s} {fmt_num(m['calinski_harabasz'], 1):>10}   (càng cao càng tốt)")

    summary = rfm.groupby("Cluster").agg(
        Recency_mean=("Recency", "mean"),
        Frequency_mean=("Frequency", "mean"),
        Monetary_mean=("Monetary", "mean"),
        SoLuong=("CUSTOMER_ID", "count"),
    ).round(2)
    summary["TyLe(%)"] = (summary["SoLuong"] / summary["SoLuong"].sum() * 100).round(1)

    # Đặt tên cụm tự động theo thứ hạng R (thấp=tốt), F, M (cao=tốt) — không phụ thuộc dataset
    rank_r = summary["Recency_mean"].rank(ascending=True)
    rank_f = summary["Frequency_mean"].rank(ascending=False)
    rank_m = summary["Monetary_mean"].rank(ascending=False)
    order = (rank_r + rank_f + rank_m).sort_values().index.tolist()

    if k == 4:
        names = ["Best Customers", "Loyal Customers", "Potential Customers", "Lost Customers"]
    else:
        names = [f"Nhóm hạng {i+1}" for i in range(k)]
    label_map = dict(zip(order, names))

    rfm["Cluster_Name"] = rfm["Cluster"].map(label_map)
    summary["Ten_cum"] = summary.index.map(label_map)

    print("\n  Đặc điểm từng cụm (đã sắp theo mức độ giá trị, cụm tốt nhất lên trước):")
    disp = summary.loc[order, ["Ten_cum", "Recency_mean", "Frequency_mean", "Monetary_mean", "SoLuong", "TyLe(%)"]]
    header = f"  {'Tên cụm':<22}{'R TB (ngày)':>13}{'F TB (lần)':>12}{'M TB':>16}{'Số KH':>10}{'Tỷ lệ':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in disp.iterrows():
        print(f"  {r['Ten_cum']:<22}{fmt_num(r['Recency_mean'], 1):>13}{fmt_num(r['Frequency_mean'], 1):>12}"
              f"{fmt_num(r['Monetary_mean'], 0):>16}{fmt_num(r['SoLuong']):>10}{r['TyLe(%)']:>8.1f}%")

    summary.to_csv(f"{OUTPUT_DIR}/cluster_summary.csv")
    rfm.to_csv(f"{OUTPUT_DIR}/rfm_with_clusters.csv", index=False)
    print(f"\n  Đã lưu bảng kết quả cụm vào: {OUTPUT_DIR}/rfm_with_clusters.csv")

    return rfm


# ====================================================================
# GIAI ĐOẠN 6: TRỰC QUAN HÓA & ĐỐI CHIẾU VỚI NHÃN CÓ SẴN
# ====================================================================
def stage6_visualize_and_validate(rfm: pd.DataFrame, labels: list):
    print_section("GIAI ĐOẠN 6: TRỰC QUAN HÓA & ĐỐI CHIẾU VỚI NHÃN CÓ SẴN")

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="Cluster_Name", palette="tab10", alpha=0.6)
    plt.title("Phân cụm khách hàng: Recency vs Monetary")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/scatter_recency_monetary.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    order = rfm.groupby("Cluster_Name")["Monetary"].mean().sort_values(ascending=False).index
    sns.barplot(data=rfm, x="Cluster_Name", y="Monetary", order=order, estimator=np.mean, errorbar=None)
    plt.title("Giá trị Monetary trung bình theo từng phân khúc")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/bar_monetary_by_cluster.png", dpi=150)
    plt.close()

    print(f"Đã lưu biểu đồ vào thư mục: {OUTPUT_DIR}/")

    for col in labels:
        print(f"\n[Đối chiếu] Cluster_Name vs {col} (nhãn có sẵn):")
        if pd.api.types.is_numeric_dtype(rfm[col]):
            print(rfm.groupby("Cluster_Name")[col].mean().round(2))
        else:
            print(pd.crosstab(rfm["Cluster_Name"], rfm[col]))


# ====================================================================
# GIAI ĐOẠN 7: ĐỀ XUẤT CHIẾN LƯỢC MARKETING
# ====================================================================
STRATEGY_MAP = {
    "Best Customers": "Chương trình VIP, ưu đãi độc quyền, quà tặng cảm ơn, mời trải nghiệm sản phẩm mới sớm.",
    "Loyal Customers": "Chương trình tích điểm, bán chéo (cross-sell), ưu đãi theo mùa để khuyến khích mua lặp lại.",
    "Potential Customers": "Quảng cáo cá nhân hóa, ưu đãi dùng thử, remarketing qua email/mạng xã hội.",
    "Lost Customers": "Khảo sát lý do rời bỏ, chiến dịch hoàn tiền/giảm giá mạnh để tái kích hoạt.",
}

def stage7_marketing_recommendation(rfm: pd.DataFrame):
    print_section("GIAI ĐOẠN 7: ĐỀ XUẤT CHIẾN LƯỢC MARKETING THEO TỪNG PHÂN KHÚC")

    order = rfm.groupby("Cluster_Name")["Monetary"].mean().sort_values(ascending=False).index
    for name in order:
        n = (rfm["Cluster_Name"] == name).sum()
        pct = n / len(rfm) * 100
        strategy = STRATEGY_MAP.get(name, "Cần phân tích thêm để đề xuất chiến lược phù hợp.")
        print(f"\n  • {name}  —  {fmt_num(n)} khách hàng ({pct:.1f}% tổng số)")
        print(f"    Chiến lược đề xuất: {strategy}")


# ====================================================================
# CHẠY TOÀN BỘ PIPELINE
# ====================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    resolved, labels = stage0_resolve_columns(DATA_PATH, COLUMN_MAP, EXTRA_LABEL_COLUMNS)
    df = stage1_load_and_explore(DATA_PATH, resolved, labels)
    df = stage2_clean_data(df)
    rfm = stage3_compute_rfm(df, labels)
    rfm_scaled, best_k = stage4_scale_and_find_k(rfm)
    chosen_k = FINAL_K if FINAL_K is not None else best_k
    print(f"\n  >>> Số cụm được chọn: k = {chosen_k}" + (" (tự động từ Silhouette)" if FINAL_K is None else " (cố định từ config)"))

    kmeans_labels = None
    if RUN_ALGORITHM_COMPARISON:
        _, _, kmeans_labels = stage4b_compare_algorithms(rfm_scaled, k=chosen_k)

    rfm = stage5_train_kmeans(rfm, rfm_scaled, k=chosen_k, precomputed_labels=kmeans_labels)
    stage6_visualize_and_validate(rfm, labels)
    stage7_marketing_recommendation(rfm)

    print("\n" + "=" * 70)
    print("HOÀN TẤT PIPELINE. Xem toàn bộ kết quả trong thư mục:", OUTPUT_DIR)
    print("=" * 70)
