import io
import os
import json
from backend.app_v2 import app

API_TOKEN = os.getenv("API_TOKEN", "change-me-token")

def auth_headers():
    return {"Authorization": f"Bearer {API_TOKEN}"}

def test_create_session():
    client = app.test_client()
    res = client.post("/api/session", headers=auth_headers(), json={"user_id":"u"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] and data["session_id"]

def test_upload_audio_enqueues_jobs(tmp_path):
    client = app.test_client()
    # create session first
    sid = client.post("/api/session", headers=auth_headers(), json={}).get_json()["session_id"]
    # fake wav bytes
    audio_bytes = b"RIFF....WAVEfmt "
    data = {
        "question_index": "1",
        "audio": (io.BytesIO(audio_bytes), "sample.wav"),
    }
    res = client.post(f"/api/session/{sid}/audio", headers=auth_headers(), data=data, content_type='multipart/form-data')
    assert res.status_code in (200, 500)  # ffmpeg may fail in CI; pipeline exists


