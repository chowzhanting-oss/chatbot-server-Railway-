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

# ──────────────────────────────────────────────────────────────────────────────
# Environment / Config
# ──────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
# STREAMING=on -> server sends text/plain once at end (buffered for MathJax)
STREAMING_DEFAULT = os.getenv("STREAMING", "on").strip().lower() == "on"
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")  # e.g. https://moodle.yoursite.edu

app = Flask(__name__)
if FRONTEND_ORIGIN == "*":
    CORS(app)  # permissive while debugging
else:
    CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})

client = OpenAI(api_key=OPENAI_API_KEY)

# ──────────────────────────────────────────────────────────────────────────────
# System prompt: on-topic + LaTeX rules
# ──────────────────────────────────────────────────────────────────────────────
LATEX_SYSTEM = (
    "You are a patient electronics tutor for Integrated Electronics (CMOS, MOSFETs, amplifiers, threshold voltage, etc.). "
    "Default: respond briefly and clearly. "
    "Use LaTeX for math: one display block with $$...$$ for multi-line equations and \\(...\\) for inline math. "
    "Do NOT escape punctuation/brackets inside math (write = ( ) [ ] ^ _ plainly). "
    "Example: $$ I_D = \\mu_n C_{ox}\\frac{W}{L}[(V_{GS}-V_T)V_{DS}-\\frac{V_{DS}^2}{2}] $$. "
    "Provide derivations only if asked. "
    "If the question is off-topic, reply exactly: "
    "Sorry I cannot help you with that, I can only answer questions about Integrated Electronics."
)

# ──────────────────────────────────────────────────────────────────────────────
# Utilities: tiny cache + LaTeX sanitizers
# ──────────────────────────────────────────────────────────────────────────────
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

# Collapse \\mu -> \mu etc.
def _collapse_double_backslashes(s: str) -> str:
    return re.sub(r"\\\\", r"\\", s)

# Inside math only, remove stray backslashes before operators/brackets.
def _fix_overescape_in_math(math: str) -> str:
    math = math.replace(r"\left\[", r"\left[").replace(r"\right\]", r"\right]")
    math = re.sub(r"\\([=\(\)\[\]\+\-\*/\^_])", r"\1", math)
    return math

_MATH_DISPLAY = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_MATH_INLINE  = re.compile(r"\\\((.*?)\\\)", re.DOTALL)

def sanitize_latex(s: str) -> str:
    """
    Make model output friendlier to MathJax:
      • collapse double backslashes globally,
      • fix over-escaped punctuation/brackets inside $$...$$ and \(...\).
    """
    s = _collapse_double_backslashes(s)
    def _fix_display(m): return "$$" + _fix_overescape_in_math(m.group(1)) + "$$"
    def _fix_inline(m):  return r"\(" + _fix_overescape_in_math(m.group(1)) + r"\)"
    s = _MATH_DISPLAY.sub(_fix_display, s)
    s = _MATH_INLINE.sub(_fix_inline, s)
    return s

# Always include CORS headers (helps with OPTIONS / preflight)
@app.after_request
def add_cors_headers(resp):
    resp.headers.setdefault("Access-Control-Allow-Origin", "*" if FRONTEND_ORIGIN == "*" else FRONTEND_ORIGIN)
    resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization")
    resp.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return resp

@app.route("/chat", methods=["OPTIONS"])
def chat_options():
    return ("", 204)

# ──────────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/ping")
def ping():
    return jsonify({"status": "ok"})

# ──────────────────────────────────────────────────────────────────────────────
# Non-streaming helper
# ──────────────────────────────────────────────────────────────────────────────
def get_full_answer(question: str) -> str:
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": LATEX_SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    return sanitize_latex(resp.output_text or "")

# ──────────────────────────────────────────────────────────────────────────────
# /chat endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("message") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'message'"}), 400

    # Non-streaming path (JSON)
    if not STREAMING_DEFAULT:
        cached = answer_cache.get(question)
        if cached:
            return jsonify({"reply": cached})
        try:
            answer = get_full_answer(question)
            answer_cache.put(question, answer)
            return jsonify({"reply": answer})
        except Exception as e:
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    # Streaming path (text/plain), but buffer for MathJax reliability
    def generate():
        parts = []
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
                        parts.append(event.delta or "")
                # After stream ends, sanitize once and send once
                final_text = sanitize_latex("".join(parts))
                yield final_text
        except Exception as e:
            yield f"\n[Stream error: {type(e).__name__}: {e}]"
        finally:
            text = "".join(parts).strip()
            if text:
                answer_cache.put(question, sanitize_latex(text))

    return Response(stream_with_context(generate()),
                    mimetype="text/plain; charset=utf-8")

# ──────────────────────────────────────────────────────────────────────────────
# Keep-alive (optional)
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Local dev runner (ignored by Gunicorn in production)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
