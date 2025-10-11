import os
import sqlite3
from datetime import datetime
from io import StringIO
import csv
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from openai import OpenAI

# -------------------- App & Config --------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Toggle streaming via env if needed: STREAMING=off|on (default: on)
STREAMING_DEFAULT = os.getenv("STREAMING", "on").lower() == "on"

# Simple in-memory cache {question: answer}
answer_cache = {}

# Use Railway's persistent volume by default
DB_PATH = os.getenv("DB_PATH", "/data/chat_history.db")

# Limit upload size for vision (tweak as needed)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB

# -------------------- DB Helpers --------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,          -- ISO8601Z
            student_id TEXT,           -- Moodle user id or 'anonymous'
            question TEXT NOT NULL,
            answer TEXT NOT NULL
        )
    """)
    # Helpful indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_ts ON chat_logs(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_student ON chat_logs(student_id)")
    conn.commit()
    conn.close()

init_db()

# -------------------- Prompting --------------------
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

# -------------------- Health --------------------
@app.get("/")
def root():
    return jsonify({"ok": True, "service": "ai-tutor", "time": datetime.utcnow().isoformat() + "Z"})

@app.get("/ping")
def ping():
    return jsonify({"status": "ok"})

# -------------------- Text Chat (streaming) --------------------
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

    # streaming path
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

# -------------------- Vision Chat (image + text) --------------------
# Accepts multipart/form-data with fields:
#   message: string (required if no image_url)
#   image:   file (optional)
#   image_url: string (optional)
#   student_id: string (optional)
def _b64_data_uri(file_storage):
    import base64, mimetypes
    data = file_storage.read()
    file_storage.seek(0)
    mime = file_storage.mimetype or mimetypes.guess_type(file_storage.filename)[0] or 'application/octet-stream'
    b64 = base64.b64encode(data).decode('ascii')
    return f"data:{mime};base64,{b64}"

def _vision_input(question, image_data_uri_or_url=None):
    parts = [{"type": "input_text", "text": question or ""}]
    if image_data_uri_or_url:
        parts.append({"type": "input_image", "image_url": image_data_uri_or_url})
    return [{"role": "user", "content": parts}]

@app.post("/vision_chat")
def vision_chat():
    msg = (request.form.get("message") or "").strip()
    student_id = (request.form.get("student_id") or "anonymous").strip() or "anonymous"
    img_file = request.files.get("image")
    img_url = (request.form.get("image_url") or "").strip()

    if not msg and not (img_file or img_url):
        return jsonify({"error": "Provide 'message' and/or an image"}), 400

    data_uri = None
    if img_file and img_file.filename:
        ext = (img_file.filename.rsplit(".", 1)[-1] or "").lower()
        if ext not in {"png", "jpg", "jpeg", "webp"}:
            return jsonify({"error": "Unsupported image type"}), 400
        data_uri = _b64_data_uri(img_file)
    elif img_url:
        data_uri = img_url  # let the model fetch it

    if not STREAMING_DEFAULT:
        resp = client.responses.create(
            model="gpt-5-mini",  # vision-capable in your account
            input=_vision_input(msg, data_uri)
        )
        answer = resp.output_text or ""
        log_chat(student_id, f"[vision] {msg}", answer)
        return jsonify({"reply": answer})

    def generate():
        collected = []
        try:
            with client.responses.stream(
                model="gpt-5-mini",
                input=_vision_input(msg, data_uri),
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        chunk = event.delta
                        collected.append(chunk)
                        yield chunk
                stream.close()
            full = "".join(collected)
            log_chat(student_id, f"[vision] {msg}", full)
        except Exception:
            # Fallback to text-only
            full = non_streaming_answer(msg or "[image]")
            log_chat(student_id, f"[vision-fallback] {msg}", full)
            yield full

    return Response(generate(), mimetype="text/plain")

# -------------------- Admin Retrieval --------------------
# Protect with a simple admin token header: X-Admin-Token: <token>
def require_admin(req) -> bool:
    token = req.headers.get("X-Admin-Token", "")
    return token == os.getenv("ADMIN_TOKEN", "")

@app.get("/admin/logs")
def admin_logs():
    if not require_admin(request):
        return jsonify({"error": "unauthorized"}), 401

    q = (request.args.get("q") or "").strip()
    student = (request.args.get("student_id") or "").strip()
    since = (request.args.get("since") or "").strip()
    until = (request.args.get("until") or "").strip()
    limit = max(1, min(int(request.args.get("limit", 50)), 500))
    offset = max(0, int(request.args.get("offset", 0)))

    where = []
    args = []
    if student:
        where.append("student_id = ?"); args.append(student)
    if q:
        where.append("(question LIKE ? OR answer LIKE ?)")
        args.extend([f"%{q}%", f"%{q}%"])
    if since:
        where.append("ts >= ?"); args.append(since)
    if until:
        where.append("ts <= ?"); args.append(until)

    sql_where = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
        SELECT id, ts, student_id, question, answer
        FROM chat_logs
        {sql_where}
        ORDER BY ts DESC
        LIMIT ? OFFSET ?
    """
    args_page = args + [limit, offset]

    conn = get_db()
    rows = [dict(r) for r in conn.execute(sql, args_page).fetchall()]
    total = conn.execute(f"SELECT COUNT(*) AS c FROM chat_logs {sql_where}", args).fetchone()["c"]
    conn.close()

    return jsonify({"total": total, "limit": limit, "offset": offset, "items": rows})

@app.get("/admin/logs/export.csv")
def admin_logs_export():
    if not require_admin(request):
        return jsonify({"error": "unauthorized"}), 401

    q = (request.args.get("q") or "").strip()
    student = (request.args.get("student_id") or "").strip()
    since = (request.args.get("since") or "").strip()
    until = (request.args.get("until") or "").strip()

    where = []
    args = []
    if student:
        where.append("student_id = ?"); args.append(student)
    if q:
        where.append("(question LIKE ? OR answer LIKE ?)")
        args.extend([f"%{q}%", f"%{q}%"])
    if since:
        where.append("ts >= ?"); args.append(since)
    if until:
        where.append("ts <= ?"); args.append(until)

    sql_where = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
        SELECT ts, student_id, question, answer
        FROM chat_logs
        {sql_where}
        ORDER BY ts DESC
    """

    conn = get_db()
    cur = conn.execute(sql, args)

    si = StringIO()
    w = csv.writer(si)
    w.writerow(["ts", "student_id", "question", "answer"])
    for r in cur:
        w.writerow([r["ts"], r["student_id"], r["question"], r["answer"]])
    conn.close()

    return Response(
        si.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=chat_logs.csv"}
    )

# -------------------- Run --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))  # Railway injects PORT
    app.run(host="0.0.0.0", port=port, debug=False)
