# HƯỚNG DẪN VẬN HÀNH HỆ THỐNG AI PHÂN TÍCH & GIỮ CHÂN KHÁCH HÀNG
## (AI Customer Intelligence & Retention System 2.0)

Hệ thống được phát triển theo kiến trúc chuẩn Clean Architecture, nâng cấp toàn diện từ mô hình Jupyter Notebook đơn lẻ thành **Hệ sinh thái AI Đa nhiệm**:
1. **Tầng Dữ liệu & ETL**: Tự động dò encoding, chuẩn hóa cột, xử lý bất thường mức khách hàng (Winsorization P99), kiểm tra chất lượng mã giao dịch.
2. **Tầng AI Đa nhiệm**:
   - *Descriptive AI*: Phân cụm RFM (K-Means, GMM, FCM, DBSCAN, Spectral, Mean Shift), tự động chọn số cụm $k$ (Silhouette & Elbow), gán nhãn hồ sơ kinh doanh (*Best Customers*, *At-Risk Customers*, *Lost Customers*...).
   - *Predictive AI*: Mô hình máy học Gradient Boosting dự báo rủi ro rời bỏ (Churn Risk) và Giá trị mua sắm tương lai (CLV).
   - *Generative AI Copilot*: Trợ lý AI Marketing Studio (Gemini 2.5 Flash / Hybrid Offline Engine) tự động lập kế hoạch chiến dịch, phân bổ ngân sách, soạn thảo nội dung tin nhắn/email cá nhân hóa.
3. **Tầng Backend API (FastAPI)**: RESTful API đầy đủ kèm tài liệu tương tác Swagger UI.
4. **Tầng Giao diện Người dùng (Streamlit Dashboard)**: Giao diện web tương tác trực quan 3D Plotly, tra cứu Customer 360, Marketing Studio.
5. **Tầng MLOps & Báo cáo**: Quản lý phiên Model Registry, xuất file báo cáo Word (`.docx`) tự động.

---

## 1. Cài đặt Môi trường

Yêu cầu: Python 3.10 trở lên.

```bash
# Cài đặt tất cả thư viện cần thiết
pip install -r requirements.txt
```

---

## 2. Hướng dẫn Khởi chạy Hệ thống

### Cách 1: Khởi chạy Giao diện Web Dashboard (Streamlit)
Đây là giao diện chính để demo trực tiếp trước giảng viên và hội đồng:

```bash
streamlit run webapp/app.py
```
> Trình duyệt sẽ tự động mở tại địa chỉ: `http://localhost:8501`

### Cách 2: Khởi chạy REST API Backend (FastAPI)
Nếu cần kết nối API với hệ thống bên ngoài (POS, Web, Mobile App):

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
> - Trang tài liệu tương tác Swagger UI: `http://localhost:8000/docs`
> - Trang kiểm tra sức khỏe hệ thống: `http://localhost:8000/health`

### Cách 3: Huấn luyện qua dòng lệnh (CLI Pipeline)
```bash
python -m src.pipeline
```

### Cách 4: Chạy toàn bộ hệ thống bằng Docker Compose
```bash
docker compose up --build
```

---

## 3. Cấu hình Khóa Google Gemini API (Tùy chọn)

Hệ thống được thiết kế theo cơ chế **Hybrid Thông minh**:
- **Mặc định (Chưa có API Key)**: Hệ thống chạy trên **Offline Rule-based Copilot**, tự động trích xuất số liệu thật của phân khúc để tạo chiến dịch marketing, bảng phân bổ ngân sách, kịch bản tin nhắn hoàn chỉnh mà không cần mạng Internet.
- **Khi có API Key**:
  1. Lấy khóa API miễn phí tại: [Google AI Studio](https://aistudio.google.com/)
  2. Tạo file `.env` ở thư mục gốc (hoặc copy từ `.env.example`):
     ```ini
     GEMINI_API_KEY=AIzaSy...your_key_here...
     ```
  3. Khởi động lại ứng dụng, hệ thống sẽ tự động kích hoạt model `gemini-2.5-flash` để mở rộng khả năng sáng tạo thông điệp marketing.

---

## 4. Kịch bản Trình diễn Demo Chuẩn cho Giảng viên

Khi báo cáo trước giảng viên hướng dẫn hoặc hội đồng, bạn nên thực hiện theo 5 bước sau:

1. **Bước 1 — Mở Dashboard (`Tab 1: Tổng quan Kinh doanh`)**:
   - Trình bày các thẻ chỉ số KPI: Tổng số khách hàng, tổng doanh thu, doanh thu trung bình mỗi khách, tần suất mua sắm và tỷ lệ khách hàng còn active trong 90 ngày.
   - Chỉ ra biểu đồ phân bổ: Nhóm nào đóng góp tỷ trọng doanh thu cao nhất (ví dụ: nhóm At-Risk hoặc Best Customers).

2. **Bước 2 — Phân tích Không gian 3D (`Tab 2: Không gian Phân cụm RFM 3D`)**:
   - Dùng chuột xoay không gian 3 chiều Recency × Frequency × Monetary.
   - Giải thích vì sao K-Means được lựa chọn thông qua **Bảng đối sánh 6 thuật toán** (K-Means, GMM, FCM, DBSCAN, Spectral, Mean Shift) với 4 chỉ số Silhouette, Davies-Bouldin, Calinski-Harabasz và Dunn index trên cùng một mẫu chuẩn hóa.

3. **Bước 3 — Tra cứu Hồ sơ Khách hàng 360 (`Tab 3: Hồ sơ Khách hàng 360`)**:
   - Nhập một mã khách hàng cụ thể (ví dụ: `C00000001`).
   - Giới thiệu thẻ Percentile: Khách này thuộc top mấy % về chi tiêu và độ tích cực.
   - Trình bày mục **"Next Best Action"**: Hành động gợi ý cụ thể, kênh liên hệ và tin nhắn mẫu được cá nhân hóa cho riêng khách hàng này.

4. **Bước 4 — AI Marketing Studio (`Tab 4: AI Marketing Studio`)**:
   - Chọn phân khúc mục tiêu (ví dụ: *At-Risk Customers*), chọn mục tiêu chiến dịch (*Win-back*), ngân sách (10.000.000 VND), kênh (*Zalo OA* hoặc *Email*).
   - Nhấn nút **"Sinh Chiến dịch bằng AI"**: Trình diễn kế hoạch tiếp thị chi tiết 5 phần gồm: Phân tích phân khúc, ưu đãi đề xuất, lịch triển khai 4 tuần, tin nhắn mẫu và bảng ước tính KPI doanh thu.

5. **Bước 5 — Quản lý & Xuất Báo cáo Word (`Tab 5: Quản lý Mô hình & Báo cáo`)**:
   - Nhấn nút **"Tạo & Tải về Báo cáo Word"**.
   - Mở file `.docx` vừa tạo để chứng minh hệ thống có khả năng xuất bản tài liệu báo cáo học thuật tự động cho ban giám đốc/hội đồng.

---

## 5. Chạy Kiểm thử Tự động (Automated Tests)

Kiểm tra toàn bộ hệ sinh thái với 14 bài kiểm thử unit & integration:

```bash
pytest tests/ -v
```
Toàn bộ các module Ingestion, RFM Engine, Clustering Evaluation, Segmentation và FastAPI endpoints đều đã được kiểm định tự động đạt 100% Passed.
