# Khởi động Backend Server

## Vấn đề thường gặp
Lỗi "TypeError: Failed to fetch" xảy ra khi frontend không thể kết nối với backend server.

## Cách khởi động Backend

### Phương pháp 1: Sử dụng script tự động (Khuyến nghị)
1. **Windows Batch**: Chạy `start_backend.bat` (double-click)
2. **PowerShell**: Chạy `start_backend.ps1` (right-click → Run with PowerShell)

### Phương pháp 2: Khởi động thủ công
1. Mở terminal/command prompt
2. Di chuyển đến thư mục backend: `cd backend`
3. Chạy lệnh: `python run.py`

## Kiểm tra Backend
Backend sẽ chạy trên `http://localhost:5001`

Để test API:
```bash
curl http://localhost:5001/api/health
```

## Lưu ý
- Đảm bảo Python và các dependencies đã được cài đặt
- Backend cần chạy trước khi mở frontend
- Nếu gặp lỗi, kiểm tra file `backend.log` để debug

## Dừng Backend
Nhấn `Ctrl+C` trong terminal đang chạy backend
