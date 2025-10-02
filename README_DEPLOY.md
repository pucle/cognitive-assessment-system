## Triển khai demo backend lên web host (Docker)

### Yêu cầu
- Docker + Docker Compose
- Tạo `backend/config.env` với API keys cần thiết:
  - `OPENAI_API_KEY`
  - `GEMINI_API_KEY`

### Build & Run (Production)
```bash
cd deploy
docker compose -f docker-compose.prod.yml up -d --build
```

- API backend: `http://<HOST>:8000`
- Nginx reverse proxy: `http://<HOST>/`

### Kiểm tra health
```bash
curl http://<HOST>:8000/api/health
```

### Xem logs
```bash
docker compose -f deploy/docker-compose.prod.yml logs -f backend
```

### Cập nhật code
```bash
git pull
cd deploy
docker compose -f docker-compose.prod.yml up -d --build
```

### Ghi chú
- Chạy bằng gunicorn phục vụ `backend.app:app`
- Image có sẵn `ffmpeg` và `libsndfile`
- Tùy chỉnh Nginx/certbot nếu cần domain + HTTPS


