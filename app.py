# app.py
import os
import re
import time
import threading
from collections import OrderedDict
from typing import Optional

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI

# -------------------------
# Environment / Config
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
STREAMING_DEFAULT = os.getenv("STREAMING", "on").strip().lower() == "on"
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")  # e.g. https://your-moodle.site

app = Flask(__name__)

# CORS (lock to your Moodle origin in prod; "*" is okay while debugging)
if FRONTEND_ORIGIN == "*":
    CORS(app)
else:
    CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# System prompt (stay on-topic + LaTeX)
# -------------------------
LATEX_SYSTEM = (
    "You are a patient electronics tutor for Integrated Electronics (CMOS, MOSFETs, amplifiers, Vth, etc.). "
    "Default: respond briefly and clearly, using short paragraphs or compact bullet points. "
    "Use LaTeX for math with display $$...$$ and inline \\(...\\). "
    "Example display: $$ I_D = \\mu_n C_{ox}\\frac{W}{L}[(V_{GS}-V_{TH})V_{DS}-\\frac{V_{DS}^2}{2}] $$. "
    "Show derivations only if the user asks to 'explain more' or 'show steps'. "
    "If the question is off-topic, reply exactly: "
    "Sorry I cannot help you with that, I can only answer questions about Integrated Electronics."
)

# -------------------------
# Utilities
# -------------------------
def normalize_latex_backslashes(s: str) -> str:
    """
    Collapse double backslashes (\\mu -> \mu, \\phi -> \phi, etc.) so MathJax parses correctly.
    Keeps \[ \] usable for display math; linebreak \\ becomes \ (ignored by MathJax).
    """
    return re.sub(r"\\\\", r"\\", s)

class LRU(OrderedDict):
    def __init__(self, maxsize=64):
        super().__init__()
        self.maxsize = maxsize
    def get(self, k) -> Optional[str]:
        if k in self:
            v = super().pop(k)
            super().__setitem__(k, v)
            return v
        return None
    def put(self, k, v):
        if k in self:
            super().pop(k)
        elif len(self) >= self.maxsize:
            self.popitem(last=False)
        super().__setitem__(k, v)

answer_cache = LRU(maxsize=64)

# Always include permissive CORS headers (handy for OPTIONS/preflight)
@app.after_request
def add_cors_headers(resp):
    resp.headers.setdefault("Access-Control-Allow-Origin", "*" if FRONTEND_ORIGIN == "*" else FRONTEND_ORIGIN)
    resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization")
    resp.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return resp

@app.route("/chat", methods=["OPTIONS"])
def chat_options():
    return ("", 204)

# -------------------------
# Health
# -------------------------
@app.get("/ping")
def ping():
    return jsonify({"status": "ok"})

# -------------------------
# Core non-streaming call
# -------------------------
def get_full_answer(question: str) -> str:
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": LATEX_SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    text = resp.output_text or ""
    return normalize_latex_backslashes(text)

# -------------------------
# /chat endpoint
# -------------------------
@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("message") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'message'"}), 400

    # Cache hit (non-streaming path only, to keep streaming simple)
    if not STREAMING_DEFAULT:
        cached = answer_cache.get(question)
        if cached:
            return jsonify({"reply": cached})

    # Non-streaming
    if not STREAMING_DEFAULT:
        try:
            answer = get_full_answer(question)
            answer_cache.put(question, answer)
            return jsonify({"reply": answer})
        except Exception as e:
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    # Streaming (text/plain) — normalize each chunk before yielding
    def generate():
        full_text = []
        try:
            with client.responses.stream(
                model=MODEL,
                input=[
                    {"role": "system", "content": LATEX_SYSTEM},
                    {"role": "user", "content": question},
                ],
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        chunk = event.delta or ""
                        chunk = normalize_latex_backslashes(chunk)
                        full_text.append(chunk)
                        yield chunk
                # (You could inspect other events here if needed)
        except Exception as e:
            yield f"\n[Stream error: {type(e).__name__}: {e}]"
        finally:
            text = "".join(full_text).strip()
            if text:
                answer_cache.put(question, text)

    return Response(stream_with_context(generate()),
                    mimetype="text/plain; charset=utf-8")

# -------------------------
# Keep-alive (optional)
# -------------------------
def keep_alive():
    url = os.getenv("SELF_PING_URL")  # e.g., https://your-app.up.railway.app
    if not url:
        return
    try:
        import requests
    except Exception:
        return
    while True:
        try:
            requests.get(url + "/ping", timeout=5)
        except Exception:
            pass
        time.sleep(600)

if os.getenv("SELF_PING_URL"):
    threading.Thread(target=keep_alive, daemon=True).start()

# -------------------------
# Local dev runner (ignored by Gunicorn)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
