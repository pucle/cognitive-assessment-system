#!/usr/bin/env python3
"""
Script quản lý recordings cho Cognitive Assessment System
Chức năng:
- Backup recordings theo ngày
- Cleanup recordings cũ (> 30 ngày)
- Report disk usage
"""

import os
import shutil
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json

def get_storage_path():
    """Lấy đường dẫn storage từ environment hoặc default"""
    return os.getenv('STORAGE_PATH', './recordings')

def get_backup_path():
    """Lấy đường dẫn backup"""
    return os.getenv('BACKUP_PATH', './backups')

def create_backup(storage_path, backup_path):
    """Tạo backup của recordings"""
    print("🔄 Đang tạo backup recordings...")

    # Tạo thư mục backup theo ngày
    today = datetime.now().strftime('%Y-%m-%d')
    backup_dir = os.path.join(backup_path, f"recordings_backup_{today}")

    try:
        # Copy toàn bộ thư mục recordings
        if os.path.exists(storage_path):
            shutil.copytree(storage_path, backup_dir, dirs_exist_ok=True)
            print(f"✅ Backup thành công: {backup_dir}")
            return True
        else:
            print("⚠️  Không tìm thấy thư mục recordings để backup")
            return False
    except Exception as e:
        print(f"❌ Lỗi khi backup: {e}")
        return False

def cleanup_old_recordings(storage_path, days_old=30):
    """Xóa recordings cũ hơn số ngày chỉ định"""
    print(f"🧹 Đang dọn dẹp recordings cũ hơn {days_old} ngày...")

    cutoff_date = datetime.now() - timedelta(days=days_old)
    deleted_count = 0
    deleted_size = 0

    if not os.path.exists(storage_path):
        print("⚠️  Không tìm thấy thư mục recordings")
        return 0, 0

    # Duyệt qua tất cả session directories
    for session_dir in os.listdir(storage_path):
        session_path = os.path.join(storage_path, session_dir)

        if not os.path.isdir(session_path):
            continue

        # Duyệt qua các file trong session
        for filename in os.listdir(session_path):
            if not filename.endswith('.wav'):
                continue

            file_path = os.path.join(session_path, filename)

            # Lấy thời gian tạo file
            try:
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))

                if file_time < cutoff_date:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    deleted_count += 1
                    deleted_size += file_size
                    print(f"🗑️  Đã xóa: {file_path}")
            except Exception as e:
                print(f"⚠️  Lỗi khi xử lý file {file_path}: {e}")

    print(f"✅ Đã xóa {deleted_count} file, tiết kiệm {deleted_size / (1024*1024):.2f} MB")
    return deleted_count, deleted_size

def get_disk_usage(storage_path):
    """Tính toán disk usage của recordings"""
    print("📊 Đang tính toán disk usage...")

    total_files = 0
    total_size = 0
    sessions = {}

    if not os.path.exists(storage_path):
        return {"total_files": 0, "total_size_mb": 0, "sessions": {}}

    # Duyệt qua tất cả session directories
    for session_dir in os.listdir(storage_path):
        session_path = os.path.join(storage_path, session_dir)

        if not os.path.isdir(session_path):
            continue

        session_files = 0
        session_size = 0

        # Duyệt qua các file trong session
        for filename in os.listdir(session_path):
            if not filename.endswith('.wav'):
                continue

            file_path = os.path.join(session_path, filename)
            try:
                file_size = os.path.getsize(file_path)
                session_files += 1
                session_size += file_size
            except Exception as e:
                print(f"⚠️  Lỗi khi đọc file {file_path}: {e}")

        sessions[session_dir] = {
            "files": session_files,
            "size_mb": session_size / (1024 * 1024)
        }

        total_files += session_files
        total_size += session_size

    return {
        "total_files": total_files,
        "total_size_mb": total_size / (1024 * 1024),
        "sessions": sessions,
        "timestamp": datetime.now().isoformat()
    }

def save_report(report_data, output_file=None):
    """Lưu report ra file JSON"""
    if not output_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"recordings_report_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"📄 Report đã lưu: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Quản lý recordings cho Cognitive Assessment')
    parser.add_argument('--backup', action='store_true', help='Tạo backup recordings')
    parser.add_argument('--cleanup', action='store_true', help='Dọn dẹp recordings cũ')
    parser.add_argument('--days', type=int, default=30, help='Số ngày để giữ recordings (default: 30)')
    parser.add_argument('--report', action='store_true', help='Tạo report disk usage')
    parser.add_argument('--all', action='store_true', help='Chạy tất cả: backup + cleanup + report')

    args = parser.parse_args()

    storage_path = get_storage_path()
    backup_path = get_backup_path()

    print("🎵 Cognitive Assessment - Recordings Manager")
    print(f"📁 Storage path: {storage_path}")
    print(f"💾 Backup path: {backup_path}")
    print("-" * 50)

    # Tạo thư mục backup nếu chưa có
    os.makedirs(backup_path, exist_ok=True)

    if args.all or args.backup:
        print("\n1. TẠO BACKUP")
        create_backup(storage_path, backup_path)

    if args.all or args.cleanup:
        print("\n2. DỌN DẸP RECORDINGS CŨ")
        cleanup_old_recordings(storage_path, args.days)

    if args.all or args.report:
        print("\n3. TẠO REPORT")
        report = get_disk_usage(storage_path)
        print(f"📊 Tổng số file: {report['total_files']}")
        print(f"💾 Tổng dung lượng: {report['total_size_mb']:.2f} MB")
        print(f"📂 Số sessions: {len(report['sessions'])}")

        save_report(report)

    print("\n✅ Hoàn thành!")

if __name__ == "__main__":
    main()
