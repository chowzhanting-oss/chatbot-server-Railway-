# app.py
import os
import re
import time
import json
import threading
from collections import OrderedDict
from typing import Optional, Tuple

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Environment / Config
# ──────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Tip: gpt-4o-mini tends to support JSON mode broadly; keep your default if you like.
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
STREAMING_DEFAULT = os.getenv("STREAMING", "on").strip().lower() == "on"
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")  # e.g. https://moodle.yoursite.edu

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

def extract_json_block(text: str) -> str:
    """
    Try to pull the first valid-looking JSON object from a text blob.
    """
    if not text:
        return ""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass
    return ""

def responses_result_to_text(resp) -> str:
    # Prefer structured content path
    try:
        return resp.output[0].content[0].text
    except Exception:
        pass
    # Fallback to flat text
    try:
        return resp.output_text or ""
    except Exception:
        return ""

def call_llm_json(system_prompt: str, user_messages: list, model: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Try to get JSON via:
      1) Responses API with response_format (if supported)
      2) Responses API without response_format
      3) Chat Completions (with/without response_format)
    Returns: (json_obj or None, raw_text or None)
    """
    # 1) Responses with JSON mode
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "system", "content": system_prompt}] + user_messages,
            response_format={"type": "json_object"},
        )
        text = responses_result_to_text(resp)
        try:
            return json.loads(text), text
        except Exception:
            # maybe the model returned extra text — try block extraction
            block = extract_json_block(text)
            if block:
                return json.loads(block), text
            return None, text
    except TypeError:
        # SDK doesn't accept response_format here; continue
        pass
    except Exception as e:
        # Other runtime errors; keep going to fallback
        last_text = getattr(e, "message", str(e))
        # don't return yet; try fallbacks

    # 2) Responses without JSON mode
    try:
        # Strengthen the instruction to start with "{" and end with "}"
        strengthened = user_messages + [
            {"role": "user", "content": "Output ONLY JSON. Begin with '{' and end with '}'."}
        ]
        resp = client.responses.create(
            model=model,
            input=[{"role": "system", "content": system_prompt}] + strengthened,
        )
        text = responses_result_to_text(resp)
        block = extract_json_block(text)
        if block:
            return json.loads(block), text
        return None, text
    except Exception as e:
        last_text = getattr(e, "message", str(e))

    # 3) Chat Completions fallback
    try:
        messages = [{"role": "system", "content": system_prompt}] + user_messages + [
            {"role": "user", "content": "Return ONLY JSON. Start with '{' and end with '}'."}
        ]
        # Try with response_format if supported
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or ""
            return json.loads(text), text
        except TypeError:
            # Retry without response_format
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            text = resp.choices[0].message.content or ""
            block = extract_json_block(text)
            if block:
                return json.loads(block), text
            return None, text
    except Exception as e:
        last_text = getattr(e, "message", str(e))

    # All attempts failed to parse JSON
    return None, last_text if 'last_text' in locals() else None

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
    Payload:
      { "mode":"analyze_adaptive_quiz",
        "schema":[...], "csv":"header\\nrows...", "run_label":"..." }
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

    json_obj, raw_text = call_llm_json(
        system_prompt=ANALYTICS_SYSTEM,
        user_messages=[
            {"role": "user", "content": user_message},
            {"role": "user", "content": prompt},
        ],
        model=MODEL,
    )

    if not isinstance(json_obj, dict):
        # Return a clear error to caller with whatever text we got back.
        return jsonify({
            "error": "LLM_JSON_PARSE_FAILED",
            "raw": raw_text or ""
        }), 200

    json_obj.setdefault("run_label", run_label)
    if "items" not in json_obj or not isinstance(json_obj["items"], list):
        json_obj["items"] = []

    return jsonify(json_obj)

# ──────────────────────────────────────────────────────────────────────────────
# /chat endpoint with robust mode switching
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/chat")
def chat():
    payload = get_payload_json()

    # Mode from JSON or query param
    mode = (payload.get("mode") or request.args.get("mode") or "chat").strip().lower()

    # If it looks like analytics, force analytics mode
    if mode == "chat" and (("csv" in payload) or ("schema" in payload)):
        mode = "analyze_adaptive_quiz"

    # 1) Analytics path
    if mode == "analyze_adaptive_quiz":
        return handle_analyze_adaptive_quiz(payload)

    # 2) Tutor path
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
