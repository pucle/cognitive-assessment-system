# Hướng Dẫn Triển Khai Recordings Storage

## 🎯 Phương Án Đã Chọn: Docker Volume (Dễ Hiểu Nhất)

### Tại Sao Chọn Phương Án Này?
- ✅ **Dễ hiểu**: Chỉ cần Docker, không cần tài khoản cloud
- ✅ **Tương thích**: Chạy giống nhau ở local và production
- ✅ **Bảo mật**: File được lưu riêng biệt với container
- ✅ **Backup dễ dàng**: Có thể copy thư mục recordings bất cứ lúc nào
- ✅ **Thay đổi code ít**: Chỉ cần cấu hình environment variable

---

## 🚀 Hướng Dẫn Triển Khai

### 1. Chuẩn Bị Environment Variables

```bash
# Copy file example
cp env.example .env

# Chỉnh sửa .env
DATABASE_URL=postgresql://username:password@localhost:5432/cognitive_assessment
STORAGE_PATH=./recordings
FLASK_ENV=production
NEXT_PUBLIC_API_URL=http://localhost:5001
```

### 2. Chạy Với Docker Compose

```bash
# Development
docker-compose up --build

# Production
docker-compose -f docker-compose.yml up -d --build
```

### 3. Kiểm Tra Recordings

```bash
# Vào container backend
docker exec -it cognitive-backend-1 bash

# Kiểm tra thư mục recordings
ls -la /app/recordings/
```

---

## 📁 Cấu Trúc File Recordings

```
recordings/
├── session_001/
│   ├── 1_2024-01-15T10-30-00.wav
│   ├── 2_2024-01-15T10-31-00.wav
│   └── 3_2024-01-15T10-32-00.wav
├── session_002/
│   ├── 1_2024-01-16T14-20-00.wav
│   └── 2_2024-01-16T14-21-00.wav
└── ...
```

**Giải thích:**
- `session_XXX`: ID của session assessment
- `1_2_3`: Số thứ tự câu hỏi
- `timestamp`: Thời gian ghi âm

---

## 🛠️ Quản Lý Recordings

### Sử Dụng Script manage_recordings.py

```bash
# Chạy tất cả: backup + cleanup + report
python scripts/manage_recordings.py --all

# Chỉ backup
python scripts/manage_recordings.py --backup

# Chỉ cleanup recordings cũ hơn 30 ngày
python scripts/manage_recordings.py --cleanup --days 30

# Chỉ tạo report
python scripts/manage_recordings.py --report
```

### Cron Job Tự Động (Linux/Mac)

```bash
# Mở crontab
crontab -e

# Thêm dòng này để chạy backup hàng ngày lúc 2:00 AM
0 2 * * * cd /path/to/your/app && python scripts/manage_recordings.py --all
```

### Cron Job Tự Động (Windows)

```batch
# Tạo file backup_recordings.bat
@echo off
cd /d D:\CognitiveAssessmentsystem
python scripts/manage_recordings.py --all

# Tạo scheduled task trong Task Scheduler chạy hàng ngày
```

---

## 📊 Theo Dõi Dung Lượng

### Xem Report

```bash
python scripts/manage_recordings.py --report
```

**Output mẫu:**
```json
{
  "total_files": 150,
  "total_size_mb": 450.25,
  "sessions": {
    "session_001": {"files": 3, "size_mb": 15.2},
    "session_002": {"files": 2, "size_mb": 8.5}
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

### Giám Sát Disk Usage

```bash
# Linux/Mac
du -sh recordings/

# Windows PowerShell
(Get-ChildItem recordings -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
```

---

## 🔧 Cấu Hình Nâng Cao

### 1. Thay Đổi Đường Dẫn Storage

```bash
# Trong .env
STORAGE_PATH=/mnt/external-drive/recordings
```

### 2. Thay Đổi Đường Dẫn Backup

```bash
# Trong .env
BACKUP_PATH=/mnt/backup-drive/recordings
```

### 3. Chạy Script Từ Container

```bash
# Trong docker-compose.yml, thêm volume cho scripts
volumes:
  - ./scripts:/app/scripts

# Chạy từ container
docker exec cognitive-backend-1 python /app/scripts/manage_recordings.py --all
```

---

## 🚨 Xử Lý Sự Cố

### Recordings Không Được Lưu

```bash
# Kiểm tra quyền thư mục
ls -la recordings/

# Kiểm tra logs
docker logs cognitive-backend-1

# Vào container kiểm tra
docker exec -it cognitive-backend-1 ls -la /app/recordings/
```

### Docker Volume Issues

```bash
# Xem volumes
docker volume ls

# Inspect volume
docker volume inspect cognitive_recordings_data

# Nếu cần xóa và tạo lại
docker-compose down -v
docker-compose up --build
```

---

## 📈 Mở Rộng Trong Tương Lai

### 1. Cloud Storage (Khi Cần Scale)

```python
# Thêm vào backend/services/storage.py
import boto3

def upload_to_s3(file_path, s3_key):
    s3 = boto3.client('s3')
    s3.upload_file(file_path, 'your-bucket', s3_key)
```

### 2. Database Storage

```sql
-- Thêm cột BLOB vào database
ALTER TABLE cognitive_assessment_results
ADD COLUMN audio_blob BYTEA;
```

### 3. CDN Distribution

- Sử dụng CloudFront cho AWS S3
- Tự động tạo signed URLs
- Cache recordings ở edge locations

---

## ✅ Checklist Triển Khai

- [ ] Copy `env.example` thành `.env`
- [ ] Cấu hình `DATABASE_URL`
- [ ] Chạy `docker-compose up --build`
- [ ] Test tạo assessment với audio
- [ ] Kiểm tra file recordings được tạo
- [ ] Chạy script backup
- [ ] Setup cron job tự động
- [ ] Monitor disk usage

---

## 🎯 Kết Luận

Phương án Docker Volume này:
- **Dễ triển khai**: Chỉ cần Docker và vài lệnh
- **Bảo mật**: File lưu ngoài container
- **Scalable**: Có thể mở rộng lên cloud storage
- **Maintainable**: Scripts tự động quản lý

Bạn có thể bắt đầu ngay với `docker-compose up --build`! 🚀
