import os
import sqlite3
import threading
import time
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from openai import OpenAI

# ------------------ Setup ------------------
app = Flask(__name__)
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simple in-memory cache {question: answer}
answer_cache = {}

# ------------------ DB ------------------
DB_PATH = os.getenv("DB_PATH", "chat_history.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
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

init_db()

# ------------------ Routes ------------------
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

    # 1. Check cache first
    if question in answer_cache:
        cached = answer_cache[question]
        return jsonify({"reply": cached, "cached": True})

    # 2. Stream fresh response from OpenAI
    def generate():
        collected = []
        try:
            with client.responses.stream(
                model="gpt-5-mini",
                input=[
                    {"role": "system", "content": "You are a patient tutor. Always explain step by step. Use clear formatting, line breaks, and short bullet points."},
                    {"role": "system", "content": "If off-topic, reply exactly: Sorry I cannot help you with that, I can only answer questions about Integrated Electronics."},
                    {"role": "user", "content": question},
                ],
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        chunk = event.delta
                        collected.append(chunk)
                        yield chunk
                stream.close()

            # Save the whole answer into cache & DB
            full_answer = "".join(collected)
            answer_cache[question] = full_answer

            conn = get_db()
            conn.execute(
                "INSERT INTO chat_logs (ts, student_id, question, answer) VALUES (?, ?, ?, ?)",
                (datetime.utcnow().isoformat(timespec="seconds") + "Z", student_id, question, full_answer),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            yield f"[Error: {e}]"

    return Response(generate(), mimetype="text/plain")

# ------------------ Keep-alive thread ------------------
def keep_alive():
    while True:
        try:
            with app.test_client() as c:
                c.get("/ping")
        except:
            pass
        time.sleep(600)  # every 10 minutes

threading.Thread(target=keep_alive, daemon=True).start()

# ------------------ Main ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
