import os
import sqlite3
import threading
import time
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Toggle streaming via env if needed: STREAMING=off|on  (default: on)
STREAMING_DEFAULT = os.getenv("STREAMING", "on").lower() == "on"

# Simple in-memory cache {question: answer}
answer_cache = {}

# Prefer a writable absolute path on Railway. Default keeps local dev working too.
DB_PATH = os.getenv("DB_PATH", "/data/chat_history.db")

def _ensure_db_dir_exists():
    """Make sure the folder for the DB exists (e.g., /data on Railway)."""
    db_dir = os.path.dirname(DB_PATH) or "."
    try:
        os.makedirs(db_dir, exist_ok=True)
    except Exception:
        # If it fails (e.g., permission), we still let connect() surface the error
        pass

def get_db():
    _ensure_db_dir_exists()
    # Allow use from Gunicorn gthread workers
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            student_id TEXT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# ------------ helpers ------------
LATEX_SYSTEM = (
    "You are a patient electronics tutor for Integrated Electronics. "
    "Default behavior: respond briefly and clearly using short bullet points or short paragraphs. "
    "Always format mathematical expressions using LaTeX between double dollar signs ($$ ... $$). "
    "Example: $$ I_D = \\mu_n C_{ox} \\frac{W}{L}[(V_{GS}-V_{TH})V_{DS}-\\frac{V_{DS}^2}{2}] $$. "
    "Only expand with detailed derivations if the user explicitly asks to 'explain more' or 'show derivation'. "
    "If the question is off-topic, reply exactly: "
    "Sorry I cannot help you with that, I can only answer questions about Integrated Electronics."
)

def log_chat(student_id, question, answer):
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO chat_logs (ts, student_id, question, answer) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(timespec="seconds") + "Z", student_id, question, answer),
        )
        conn.commit()
        conn.close()
    except Exception:
        # Don't crash request handling on logging failure
        pass

def non_streaming_answer(question: str) -> str:
    """Plain (non-streaming) call; returns full text."""
    resp = client.responses.create(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": LATEX_SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    return resp.output_text or ""

# ------------ routes ------------
@app.get("/ping")
def ping():
    return jsonify({"status": "ok"})

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    student_id = (data.get("student_id") or "anonymous").strip() or "anonymous"

    if not question:
        return jsonify({"error": "No message received"}), 400

    # cache
    cached = answer_cache.get(question)
    if cached:
        return jsonify({"reply": cached, "cached": True})

    # non-streaming path forced by env
    if not STREAMING_DEFAULT:
        answer = non_streaming_answer(question)
        answer_cache[question] = answer
        log_chat(student_id, question, answer)
        return jsonify({"reply": answer})

    # try streaming; if not allowed, fall back
    def generate():
        collected = []
        try:
            with client.responses.stream(
                model="gpt-5-mini",
                input=[
                    {"role": "system", "content": LATEX_SYSTEM},
                    {"role": "user", "content": question},
                ],
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        chunk = event.delta
                        collected.append(chunk)
                        yield chunk
                stream.close()

            full = "".join(collected)
            answer_cache[question] = full
            log_chat(student_id, question, full)

        except Exception:
            # streaming not available → non-stream fallback
            full = non_streaming_answer(question)
            answer_cache[question] = full
            log_chat(student_id, question, full)
            yield full

    return Response(generate(), mimetype="text/plain")

# keep-alive to reduce cold starts on free tier
def keep_alive():
    while True:
        try:
            with app.test_client() as c:
                c.get("/ping")
        except Exception:
            pass
        time.sleep(600)  # every 10 minutes

# Initialize the DB only when the app is ready to serve traffic
@app.before_first_request
def _bootstrap_db_once():
    try:
        init_db()
    except Exception:
        # Let requests still proceed; errors will show in logs if directory not writable
        pass

threading.Thread(target=keep_alive, daemon=True).start()

if __name__ == "__main__":
    # Railway provides PORT; default to 8080 for container platforms
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
