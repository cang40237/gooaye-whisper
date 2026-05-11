"""
Gooaye Whisper API - Render.com Web Service
收到 webhook 後下載 Drive 音檔，執行 faster-whisper transcription，結果存回 Drive
"""
import io
import os
import json
import tempfile
import traceback
from datetime import datetime

from flask import Flask, request, jsonify
import requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from faster_whisper import WhisperModel

app = Flask(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_PATH = "/app/secrets/google_oauth_tokens.json"
CLIENT_PATH = "/app/secrets/google_oauth_client.json"
FOLDER_ID = "1SwoScajvBc5WJZwP79T8k9Skc2cwUI2W"


def _load_json_env(key):
    val = os.environ.get(key, "")
    if val:
        return json.loads(val)
    # fallback: read from secret file
    path = f"/app/secrets/{key.lower().replace('_token','_tokens').replace('_client','_client').replace('g_','google_')}.json"
    with open(path) as f:
        return json.load(f)


def get_drive_service():
    token_data = _load_json_env("G_TOKEN")
    client_data = _load_json_env("G_CLIENT")

    # Handle nested "installed" structure
    if "installed" in client_data:
        client_data = client_data["installed"]

    creds = Credentials(
        token=token_data.get("access_token"),
        refresh_token=token_data.get("refresh_token"),
        id_token=token_data.get("id_token"),
        client_id=client_data.get("client_id"),
        client_secret=client_data.get("client_secret"),
        scopes=token_data.get("scope", "").split(),
        token_uri="https://oauth2.googleapis.com/token"
    )
    return build("drive", "v3", credentials=creds)


def refresh_credentials(creds):
    """Refresh OAuth tokens if expired."""
    if creds.expired and creds.refresh_token:
        creds.refresh(InstalledAppFlow._build_implied_scopes(creds))
        # Save refreshed token
        token_path = "/app/secrets/google_oauth_tokens.json"
        with open(token_path, "w") as f:
            json.dump({
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": creds.scopes
            }, f)
    return creds


def find_audio_file(service, filename):
    """Find a file in Drive by name within the folder."""
    results = service.files().list(
        q=f"'{FOLDER_ID}' in parents and name = '{filename}' and trashed = false",
        fields="files(id, name, mimeType)"
    ).execute()
    return results.get("files", [])


def download_file(service, file_id):
    """Download file content from Drive."""
    request_fn = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request_fn)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()


def upload_file(service, name, content, mime_type="text/plain"):
    """Upload a file to Drive folder."""
    file_metadata = {
        "name": name,
        "parents": [FOLDER_ID]
    }
    fh = io.BytesIO(content)
    media = MediaIoBaseUpload(fh, mimetype=mime_type)
    return service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id"
    ).execute()


def transcribe(audio_bytes: bytes, filename: str) -> str:
    """Run faster-whisper transcription on audio bytes."""
    
    # Write to temp file (faster-whisper supports file paths)
    suffix = ".webm" if filename.lower().endswith(".webm") else ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    try:
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, info = model.transcribe(
            tmp_path,
            language="zh",
            beam_size=5,
            vad_filter=True
        )
        
        lines = []
        for seg in segments:
            lines.append(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
        
        return "\n".join(lines)
    finally:
        os.unlink(tmp_path)


@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Webhook payload:
    {
      "filename": "EP661.mp3",
      "file_id": "Drive_file_id",
      "episode": 661
    }
    """
    try:
        payload = request.get_json()
        filename = payload.get("filename")
        file_id = payload.get("file_id")
        
        if not filename or not file_id:
            return jsonify({"error": "missing filename or file_id"}), 400
        
        print(f"[{datetime.now()}] Processing: {filename}")
        
        service = get_drive_service()
        
        # Download audio from Drive
        audio_bytes = download_file(service, file_id)
        print(f"Downloaded {len(audio_bytes) / 1024 / 1024:.1f} MB")
        
        # Transcribe
        print(f"Starting transcription...")
        transcript = transcribe(audio_bytes, filename)
        print(f"Transcription complete: {len(transcript)} chars")
        
        # Upload transcript to Drive
        base_name = filename.rsplit(".", 1)[0]
        txt_name = f"{base_name}.txt"
        txt_bytes = transcript.encode("utf-8")
        result = upload_file(service, txt_name, txt_bytes, "text/plain")
        print(f"Uploaded transcript: {txt_name} (id: {result['id']})")
        
        return jsonify({
            "success": True,
            "filename": filename,
            "transcript_id": result["id"],
            "chars": len(transcript)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Gooaye Whisper API",
        "endpoints": ["/webhook", "/health"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))