Golden Fish – Light Up Memories (Backend & Integration)

Overview

This backend provides a complete, runnable, mock-friendly infrastructure for the Golden Fish – Light Up Memories app using Flask + WebSocket, Celery + Redis queues, and local filesystem storage. It integrates three concurrent pipelines:

- GPT Realtime interviewer (websocket-driven, mockable): greets user, reads each question exactly once (TTS), listens for the answer, accepts end-of-answer via literal phrase ("done"/"đã xong") or explicit API call.
- Background ASR (Whisper / PhoWhisper, mockable): authoritative transcription written to text-records/.
- Cognitive assessment (Librosa/Praat-style features + SVM): acoustic features + MMSE prediction saved to results/.

Quick Start (mock mode)

1) Copy .env.example to .env and adjust values (leave EXTERNAL_SERVICES_MODE=mock):

   cp .env.example .env

2) Start services via Docker Compose:

   docker compose up --build

3) Health checks and testing:
- Create a session (curl examples further below)
- Upload audio to a session question
- Watch Celery logs for job execution

Key Endpoints

- POST /api/session — create session
- GET /api/session/{session_id}/question — get next question
- POST /api/session/{session_id}/audio — upload audio (multipart/form-data)
- POST /api/session/{session_id}/end_answer — force end-of-answer
- GET /api/session/{session_id}/results — session summary
- WebSocket: ws://{host}/ws/session/{session_id}
  Events: question_asked, transcript_partial, transcript_final, result_card, session_summary, error

Run locally without Docker

1) Python 3.10+
2) pip install -r requirements.txt
3) Redis running locally (default at redis:6379 in docker; set REDIS_URL for local)
4) Start Celery worker:

   celery -A backend.workers.celery_worker.celery_app worker --loglevel=INFO

5) Start Flask API:

   python backend/app_v2.py

Assumptions

- Local filesystem storage by default under ./recordings, ./text-records, ./results.
- Optional S3 adapter can be implemented later (see services/storage.py TODOs).
- SVM model exists at /models/svm_model.pkl (volume mounted in docker-compose). If /models/scaler.pkl exists, it will be used; otherwise a simple MinMax scaling is applied.

Security & privacy

- Bearer token auth for APIs. Provide API_TOKEN in .env.
- Example AES-256 at-rest encryption helper shown in services/storage.py; you can enable it for audio-at-rest with secure key/IV management.
- Default retention policy recommendations in this README (30/90/365 days). See comments/TODOs for implementing automated retention.

OpenAPI spec

See openapi/openapi.yaml for a concise Swagger snippet for the core endpoints.

Tests

- Pytest stubs in tests/ for:
  - audio upload
  - enqueue transcription
  - worker success/failure handling (mock mode)

Example curl

Create session

  curl -X POST http://localhost:8000/api/session \
    -H "Authorization: Bearer $API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"user_id":"demo-user"}'

Fetch next question

  curl -X GET http://localhost:8000/api/session/SESSION_ID/question \
    -H "Authorization: Bearer $API_TOKEN"

Upload audio (multipart)

  curl -X POST http://localhost:8000/api/session/SESSION_ID/audio \
    -H "Authorization: Bearer $API_TOKEN" \
    -F question_index=1 \
    -F audio=@/path/to/sample.wav

Mark end-of-answer

  curl -X POST http://localhost:8000/api/session/SESSION_ID/end_answer \
    -H "Authorization: Bearer $API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"question_index":1}'

Fetch results

  curl -X GET http://localhost:8000/api/session/SESSION_ID/results \
    -H "Authorization: Bearer $API_TOKEN"

Notes

- In mock mode (EXTERNAL_SERVICES_MODE=mock), external calls return deterministic responses, so the suite runs without keys.
- For real mode, configure OPENAI_API_KEY, GPT_REALTIME_MODEL, WHISPER settings in .env and set EXTERNAL_SERVICES_MODE=real.


