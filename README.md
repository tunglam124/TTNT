# AI Customer Intelligence & Retention System 2.0

> **Dự án:** Hệ thống AI Phân tích & Giữ chân Khách hàng Toàn diện (End-to-End Customer Intelligence & Retention AI System)  
> **Ngôn ngữ & Nền tảng:** Python 3.10+, FastAPI, Streamlit, Plotly, Scikit-learn, Google Gemini API

---

## 🌟 Tính Năng Nổi Bật

1. **ETL & Data Ingestion Thông Minh:**
   - Tự động nhận diện encoding file (`utf-8`, `utf-8-sig`, `cp1252`, `latin-1`).
   - Tự động khớp vai trò cột (`customer_id`, `transaction_id`, `date`, `amount`...) từ danh mục từ khóa mở rộng.
   - Phát hiện mã giao dịch dùng chung nhiều khách để chuyển đổi chế độ tính Frequency linh hoạt.
   - Xử lý ngoại lai mức khách hàng (Winsorization P99).

2. **Lõi AI Đa Nhiệm (Multi-Model AI):**
   - **Descriptive AI (Phân cụm RFM):** K-Means, Gaussian Mixture Model, Fuzzy C-Means, DBSCAN, Spectral Clustering, Mean Shift.
   - **Tối ưu $k$ tự động:** Kết hợp biểu đồ Elbow (WCSS) và chỉ số Silhouette.
   - **Gán nhãn kinh doanh tự động:** Đặt tên cụm theo hồ sơ giá trị (*Best Customers*, *At-Risk Customers*, *Lost Customers*...) không phụ thuộc vào giá trị $k$.
   - **Predictive AI:** Huấn luyện mô hình HistGradientBoosting dự báo nguy cơ rời bỏ (Churn) và Giá trị mua sắm tương lai (CLV 90 ngày) theo mốc chia thời gian (Time-split cross validation).
   - **Generative AI Marketing Copilot (Hybrid):** Trợ lý AI lập chiến dịch marketing tự động, phân bổ ngân sách, soạn thảo tin nhắn cá nhân hóa. Chạy chế độ **Offline** khi chưa có API key và kết nối **Gemini 2.5 Flash** khi có API key.

3. **Giao diện Trực quan (Streamlit Dashboard):**
   - Tổng quan KPI kinh doanh & phân bổ doanh thu.
   - Trực quan hóa không gian 3D tương tác đa chiều (Plotly 3D).
   - Tra cứu hồ sơ Customer 360 & Next Best Action.
   - AI Marketing Studio & Live Copilot Q&A.
   - Quản lý phiên Model Registry và xuất báo cáo Word (`.docx`) tự động.

4. **REST API Chuẩn Công Nghiệp (FastAPI):**
   - Tài liệu tương tác Swagger UI tại `/docs`.
   - Endpoint chấm điểm phân khúc thời gian thực cho dữ liệu mới.
   - Hỗ trợ tác vụ huấn luyện chạy ngầm (Background Tasks).

---

## 🚀 Khởi Chạy Nhanh

### 1. Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy Giao diện Dashboard (Streamlit):
```bash
streamlit run webapp/app.py
```
👉 Mở trình duyệt tại: `http://localhost:8501`

### 3. Khởi chạy REST API Backend (FastAPI):
```bash
uvicorn api.main:app --port 8000 --reload
```
👉 Mở Swagger UI tại: `http://localhost:8000/docs`

### 4. Chạy Kiểm thử:
```bash
pytest tests/ -v
```

Chi tiết tài liệu vận hành và kịch bản demo: xem [HUONG_DAN_HE_THONG_AI.md](file:///d:/project/rfm-analyst/HUONG_DAN_HE_THONG_AI.md).
