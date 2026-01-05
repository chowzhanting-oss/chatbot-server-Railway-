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
STREAMING_DEFAULT = os.getenv("STREAMING", "on").strip().lower() == "on"
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

app = Flask(__name__)
if FRONTEND_ORIGIN == "*":
    CORS(app)
else:
    CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})

client = OpenAI(api_key=OPENAI_API_KEY)

# ──────────────────────────────────────────────────────────────────────────────
# System prompt: concise, paragraph-based, no hyphens
# ──────────────────────────────────────────────────────────────────────────────
LATEX_SYSTEM = (
    "You are a patient electronics tutor for Integrated Electronics "
    "(CMOS, MOSFETs, amplifiers, threshold voltage, etc.). "
    "Be concise and cost-efficient. Use the minimum words needed to fully answer the question "
    "without omitting crucial information. "
    "Do not add background unless required. "
    "Provide derivations only if explicitly asked. "
    "Write in short, clean paragraphs. Do not use bullet points or hyphenated lists. "
    "If the question is off-topic, reply exactly: "
    "Sorry I cannot help you with that, I can only answer questions about Integrated Electronics."
)

# ──────────────────────────────────────────────────────────────────────────────
# Utilities: cache
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

# ──────────────────────────────────────────────────────────────────────────────
# LaTeX sanitizers
# ──────────────────────────────────────────────────────────────────────────────
def _collapse_command_backslashes(s: str) -> str:
    s = re.sub(r"\\\\(?=[A-Za-z\[\]])", r"\\", s)
    s = re.sub(r"\\{3,}", r"\\", s)
    return s

def _fix_overescape_in_math(math: str) -> str:
    math = math.replace(r"\left\[", r"\left[").replace(r"\right\]", r"\right]")
    math = re.sub(r"\\([=\(\)\[\]\+\-\*/\^_])", r"\1", math)
    return math

_MATH_DISPLAY = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_MATH_INLINE  = re.compile(r"\\\((.*?)\\\)", re.DOTALL)

def sanitize_latex(s: str) -> str:
    s = _collapse_command_backslashes(s)
    s = re.sub(r"\[\s*\d+(?:\.\d+)?\s*(?:pt|em|ex|mm|cm|in|bp|px)\s*\]", "", s)

    def _fix_display(m): return "$$" + _fix_overescape_in_math(m.group(1)) + "$$"
    def _fix_inline(m):  return r"\(" + _fix_overescape_in_math(m.group(1)) + r"\)"
    s = _MATH_DISPLAY.sub(_fix_display, s)
    s = _MATH_INLINE.sub(_fix_inline, s)
    return s

# ──────────────────────────────────────────────────────────────────────────────
# Text formatting: remove bullets + justify paragraphs (safe)
# ──────────────────────────────────────────────────────────────────────────────
JUSTIFY_WIDTH = 80

def _justify_paragraph(words, width):
    if len(words) == 1:
        return words[0]

    total_chars = sum(len(w) for w in words)
    gaps = len(words) - 1

    # If words alone exceed width, justification is impossible without squeezing spaces to 0.
    # Keep normal single spaces.
    if total_chars >= width or gaps <= 0:
        return " ".join(words)

    spaces_needed = width - total_chars
    space, extra = divmod(spaces_needed, gaps)

    # Guarantee at least 1 space per gap for readability
    if space <= 0:
        return " ".join(words)

    line = ""
    for i, word in enumerate(words[:-1]):
        line += word
        line += " " * (space + (1 if i < extra else 0))
    return line + words[-1]

def justify_text(text: str) -> str:
    lines = text.split("\n")
    output = []

    for line in lines:
        stripped = line.strip()

        # Preserve empty lines and math-ish lines exactly (avoid breaking MathJax)
        if not stripped or "$" in stripped or r"\(" in stripped or r"\)" in stripped:
            output.append(line)
            continue

        words = stripped.split()
        if not words:
            output.append(stripped)
            continue

        # Only justify if it's a normal paragraph line and justification won't collapse spaces
        total_chars = sum(len(w) for w in words)
        if total_chars < JUSTIFY_WIDTH and (total_chars + (len(words) - 1)) >= JUSTIFY_WIDTH:
            output.append(_justify_paragraph(words, JUSTIFY_WIDTH))
        else:
            # Keep original spacing (single spaces) for short/long lines
            output.append(" ".join(words))

    return "\n".join(output)

def format_reply(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s

    # Remove bullet / hyphen prefixes
    s = re.sub(r"(?m)^\s*(?:[-•*–—]\s+)+", "", s)

    # Normalize blank lines
    s = re.sub(r"\n{3,}", "\n\n", s)

    # Justify safely
    s = justify_text(s)

    return s.strip()

# ──────────────────────────────────────────────────────────────────────────────
# CORS
# ──────────────────────────────────────────────────────────────────────────────
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
    return format_reply(sanitize_latex(resp.output_text or ""))

# ──────────────────────────────────────────────────────────────────────────────
# /chat endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("message") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'message'"}), 400

    if not STREAMING_DEFAULT:
        cached = answer_cache.get(question)
        if cached:
            return jsonify({"reply": cached})
        answer = get_full_answer(question)
        answer_cache.put(question, answer)
        return jsonify({"reply": answer})

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
                final_text = format_reply(sanitize_latex("".join(parts)))
                yield final_text
        except Exception as e:
            yield f"\n[Stream error: {type(e).__name__}: {e}]"
        finally:
            text = "".join(parts).strip()
            if text:
                answer_cache.put(question, format_reply(sanitize_latex(text)))

    return Response(stream_with_context(generate()),
                    mimetype="text/plain; charset=utf-8")

# ──────────────────────────────────────────────────────────────────────────────
# Keep-alive
# ──────────────────────────────────────────────────────────────────────────────
def keep_alive():
    url = os.getenv("SELF_PING_URL")
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
# Local dev
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
