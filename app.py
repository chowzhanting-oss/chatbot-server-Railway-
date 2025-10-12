# app.py
import os
import time
import threading
from collections import OrderedDict
from typing import Optional

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI

# -------------------------
# Config / Environment
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
STREAMING_DEFAULT = os.getenv("STREAMING", "on").strip().lower() == "on"

# Frontend origin for CORS (set this in Railway -> Variables)
# example: https://moodle.yourschool.edu
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

app = Flask(__name__)

# CORS: lock to your Moodle origin in prod; "*" is okay while debugging
if FRONTEND_ORIGIN == "*":
    CORS(app)  # permissive
else:
    CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# System prompt (stay on-topic)
# -------------------------
LATEX_SYSTEM = (
    "You are a patient electronics tutor for Integrated Electronics (CMOS/MOSFETs, amplifiers, Vth, etc.). "
    "Default: respond briefly and clearly, using short paragraphs or tight bullet points. "
    "Use LaTeX for math with proper delimiters: display math with $$...$$ and inline math with \\(...\\). "
    "Example display: $$ I_D = \\mu_n C_{ox}\\frac{W}{L}[(V_{GS}-V_{TH})V_{DS}-\\frac{V_{DS}^2}{2}] $$. "
    "Offer derivations only if the user asks to 'explain more' or 'show steps'. "
    "If the question is off-topic, reply exactly: "
    "Sorry I cannot help you with that, I can only answer questions about Integrated Electronics."
)

# -------------------------
# Tiny LRU cache for identical questions
# -------------------------
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

# -------------------------
# Health check
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
    return resp.output_text or ""

# -------------------------
# /chat endpoint
# -------------------------
@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("message") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'message'"}), 400

    # cache hit?
    cached = answer_cache.get(question)
    if cached and not STREAMING_DEFAULT:
        return jsonify({"reply": cached})

    # choose mode
    if not STREAMING_DEFAULT:
        try:
            answer = get_full_answer(question)
            answer_cache.put(question, answer)
            return jsonify({"reply": answer})
        except Exception as e:
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    # STREAMING branch: return plain text chunks (great for chat UIs)
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
                        full_text.append(chunk)
                        # yield raw text so the browser sees $$...$$ as-is
                        yield chunk
                # (Optional) handle other event types here if you need
        except Exception as e:
            yield f"\n[Stream error: {type(e).__name__}: {e}]"
        finally:
            text = "".join(full_text).strip()
            if text:
                answer_cache.put(question, text)

    return Response(stream_with_context(generate()),
                    mimetype="text/plain; charset=utf-8")

# -------------------------
# Keep-alive pinger (prevents cold sleep on some hosts)
# -------------------------
def keep_alive():
    import requests  # only used here to ping ourselves
    url = os.getenv("SELF_PING_URL")  # set to your Railway URL to use
    if not url:
        return
    while True:
        try:
            requests.get(url + "/ping", timeout=5)
        except Exception:
            pass
        time.sleep(600)  # every 10 min

if os.getenv("SELF_PING_URL"):
    threading.Thread(target=keep_alive, daemon=True).start()

# -------------------------
# Local dev runner (ignored by Gunicorn)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # host 0.0.0.0 so phones on LAN can reach it during local tests
    app.run(host="0.0.0.0", port=port, debug=False)
