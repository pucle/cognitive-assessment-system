# Test Database Sync - Hướng dẫn kiểm tra

## Vấn đề đã được sửa

### 1. ✅ **Settings page sync với database**
- Thêm auto-refresh mỗi 30 giây
- Thêm button 🔄 để refresh thủ công
- Hiển thị loading state khi đang tải

### 2. ✅ **Greeting hiển thị đúng tên**
- Sửa logic để hiển thị "Đình Phúc" thay vì chỉ "Phúc"
- Xử lý tên có 2 từ: lấy cả 2 từ

### 3. ✅ **Database schema được cập nhật**
- Tạo bảng `users` chính
- Giữ bảng `user_reports` cho backward compatibility
- Sử dụng `text` thay vì `integer` cho age

## Cách test

### 1. **Kiểm tra Settings page**
```bash
# Mở trang settings
http://localhost:3000/settings

# Quan sát:
- Loading state khi tải
- Thông tin user được hiển thị
- Button 🔄 để refresh
- Auto-refresh mỗi 30 giây
```

### 2. **Kiểm tra Greeting**
```bash
# Mở trang cognitive assessment
http://localhost:3000/cognitive-assessment

# Quan sát:
- Greeting hiển thị "Đình Phúc" (đầy đủ)
- Không còn chỉ hiển thị "Phúc"
```

### 3. **Test Database sync**
```bash
# 1. Cập nhật thông tin trong profile
# 2. Quay lại settings page
# 3. Click button 🔄
# 4. Kiểm tra thông tin có được cập nhật không
```

### 4. **Test API endpoints**
```bash
# Test database API
curl "http://localhost:3000/api/database/user?email=dinhphuc@example.com"

# Test profile API
curl "http://localhost:3000/api/profile/user?email=dinhphuc@example.com"
```

## Cấu hình cần thiết

### 1. **Environment variables**
Tạo file `.env.local` trong `frontend`:
```env
DATABASE_URL=postgresql://username:password@host:port/database
NEON_DATABASE_URL=postgresql://username:password@host:port/database
```

### 2. **Database setup**
```sql
-- Tạo bảng users
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  age VARCHAR(10) NOT NULL,
  gender VARCHAR(20) NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  phone VARCHAR(20),
  avatar VARCHAR(500),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Thêm dữ liệu mẫu
INSERT INTO users (name, age, gender, email, phone) VALUES 
('Đình Phúc', '25', 'Nam', 'dinhphuc@example.com', '0123456789');
```

### 3. **Install dependencies**
```bash
cd frontend
npm install @neondatabase/serverless drizzle-orm
```

## Troubleshooting

### 1. **Settings không hiển thị thông tin mới**
- Kiểm tra console logs
- Click button 🔄 để refresh thủ công
- Kiểm tra database có dữ liệu mới không

### 2. **Greeting vẫn sai**
- Kiểm tra `userData.name` trong console
- Đảm bảo tên trong database là "Đình Phúc"
- Restart development server

### 3. **Database connection error**
- Kiểm tra `DATABASE_URL` trong `.env.local`
- Kiểm tra database có hoạt động không
- Kiểm tra firewall/network

## Kết quả mong đợi

Sau khi test thành công:
- ✅ Settings page hiển thị thông tin mới từ database
- ✅ Greeting hiển thị "Đình Phúc" đầy đủ
- ✅ Auto-sync hoạt động mỗi 30 giây
- ✅ Manual sync hoạt động khi click button 🔄
- ✅ Fallback system hoạt động khi database lỗi
