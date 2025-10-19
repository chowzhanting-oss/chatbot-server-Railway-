# app.py
import os
import re
import time
import json
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
# System prompts
# ──────────────────────────────────────────────────────────────────────────────
LATEX_SYSTEM = (
    "You are a patient electronics tutor for Integrated Electronics (CMOS, MOSFETs, amplifiers, threshold voltage, etc.). "
    "Default: respond briefly and clearly. "
    "Use LaTeX for math: one display block with $$...$$ for multi-line equations and \\(...\\) for inline math. "
    "Do NOT escape punctuation/brackets inside math (write = ( ) [ ] ^ _ plainly). "
    "Avoid layout directives like [6pt], [8pt], etc. "
    "Example: $$ I_D = \\mu_n C_{ox}\\frac{W}{L}[(V_{GS}-V_T)V_{DS}-\\frac{V_{DS}^2}{2}] $$. "
    "Provide derivations only if asked. "
    "If the question is off-topic, reply exactly: "
    "Sorry I cannot help you with that, I can only answer questions about Integrated Electronics."
)

ANALYTICS_SYSTEM = (
    "You are a strict learning analytics assistant. "
    "You must return a single valid JSON object matching the requested schema. "
    "Do not include any prose or commentary outside of JSON. "
    "Be conservative with risk when confidence is low."
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

def extract_json_from_responses(resp) -> str:
    try:
        return resp.output[0].content[0].text
    except Exception:
        pass
    try:
        return resp.output_text or ""
    except Exception:
        return ""

def get_payload_json() -> dict:
    """Parse JSON robustly even if Content-Type is wrong."""
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data
    raw = request.data or b""
    if raw:
        try:
            return json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception:
            pass
    return {}

# Always include CORS headers
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
# Tutor helper
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
# Analytics handler
# ──────────────────────────────────────────────────────────────────────────────
def handle_analyze_adaptive_quiz(payload: dict):
    """
    Expected:
      { "mode":"analyze_adaptive_quiz",
        "schema":[...], "csv":"header\\nrows...", "run_label":"..." }
    Returns: { "run_label": "...", "items":[ ... ] }
    """
    schema = payload.get("schema") or ["userid","username","quizname","difficultysum","standarderror","measure","timetaken"]
    csv_text = (payload.get("csv") or "").strip()
    run_label = payload.get("run_label") or f"manual_{time.strftime('%Y-%m-%d')}"
    user_message = payload.get("message") or "Analyze the CSV and return strict JSON."

    if not csv_text:
        return jsonify({"error": "Missing 'csv' content"}), 400

    schema_list = ", ".join(schema)
    prompt = f"""
You will be given a CSV with columns: {schema_list}.
Return a SINGLE valid JSON object with this exact shape:

{{
  "run_label": string,
  "items": [
    {{
      "userid": int,
      "risk_score": number,   // 0..100
      "confidence": number,   // 0..1
      "drivers": [string],    // 1-4 concise reasons
      "student_msg": string,  // short actionable message for the student
      "teacher_msg": string,  // short actionable message for the teacher
      "features": object      // echo useful per-user fields (e.g., measure, timetaken, quizname)
    }}
  ]
}}

Rules:
- Output ONLY JSON (no Markdown, no commentary).
- If data is insufficient, keep risk_score ~50 and confidence low, but still produce valid JSON.
- Aggregate rows by userid so each user appears once.

CSV data:
{csv_text}
    """.strip()

    try:
        resp = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": ANALYTICS_SYSTEM},  # bypass tutor guard
                {"role": "user", "content": user_message},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw_json_str = extract_json_from_responses(resp)
        data = json.loads(raw_json_str or "{}")
        if not isinstance(data, dict):
            data = {}
        data.setdefault("run_label", run_label)
        if "items" not in data or not isinstance(data["items"], list):
            data["items"] = []
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

# ──────────────────────────────────────────────────────────────────────────────
# /chat endpoint with robust mode switching
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/chat")
def chat():
    payload = get_payload_json()

    # Accept mode from JSON or query (fallback)
    mode = (payload.get("mode") or request.args.get("mode") or "chat").strip().lower()

    # Extra safeguard: if it looks like an analytics request (has csv/schema), force analytics mode
    if mode == "chat" and (("csv" in payload) or ("schema" in payload)):
        mode = "analyze_adaptive_quiz"

    # 1) Analytics path (never stream)
    if mode == "analyze_adaptive_quiz":
        return handle_analyze_adaptive_quiz(payload)

    # 2) Tutor path (electronics)
    question = (payload.get("message") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'message'"}), 400

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
# Local dev runner
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
