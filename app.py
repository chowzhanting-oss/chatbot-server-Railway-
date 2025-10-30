# app.py
import os, re, time, json, threading, traceback
from collections import OrderedDict
from typing import Optional, Tuple

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")   # safer default
STREAMING_DEFAULT = os.getenv("STREAMING", "on").strip().lower() == "on"
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
DEBUG_ERRORS = os.getenv("DEBUG_ERRORS", "on").strip().lower() == "on"  # show trace in JSON errors

app = Flask(__name__)
if FRONTEND_ORIGIN == "*":
    CORS(app)
else:
    CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})

client = OpenAI(api_key=OPENAI_API_KEY)

# ── System prompts ────────────────────────────────────────────────────────────
LATEX_SYSTEM = (
    "You are a patient electronics tutor for Integrated Electronics (CMOS, MOSFETs, amplifiers, threshold voltage, etc.). "
    "Default: respond briefly and clearly. Use LaTeX with $$...$$ and \\(...\\). "
    "If off-topic, reply exactly: Sorry I cannot help you with that, I can only answer questions about Integrated Electronics."
)
ANALYTICS_SYSTEM = (
    "You are a strict learning analytics assistant. "
    "Return a single valid JSON object matching the requested schema. "
    "No prose outside JSON. Be conservative with risk when confidence is low."
)

# ── Tiny cache + LaTeX cleaners (unchanged) ───────────────────────────────────
class LRU(OrderedDict):
    def __init__(self, maxsize=64): super().__init__(); self.maxsize = maxsize
    def get(self, k) -> Optional[str]:
        if k in self:
            v = super().pop(k); super().__setitem__(k, v); return v
        return None
    def put(self, k, v):
        if k in self: super().pop(k)
        elif len(self) >= self.maxsize: self.popitem(last=False)
        super().__setitem__(k, v)

answer_cache = LRU(maxsize=64)

import re as _re
def _collapse_command_backslashes(s: str) -> str:
    s = _re.sub(r"\\\\(?=[A-Za-z\[\]])", r"\\", s)
    s = _re.sub(r"\\{3,}", r"\\", s); return s
def _fix_overescape_in_math(math: str) -> str:
    math = math.replace(r"\left\[", r"\left[").replace(r"\right\]", r"\right]")
    math = _re.sub(r"\\([=\(\)\[\]\+\-\*/\^_])", r"\1", math); return math
_MATH_DISPLAY = _re.compile(r"\$\$(.*?)\$\$", _re.DOTALL)
_MATH_INLINE  = _re.compile(r"\\\((.*?)\\\)", _re.DOTALL)
def sanitize_latex(s: str) -> str:
    s = _collapse_command_backslashes(s)
    s = _re.sub(r"\[\s*\d+(?:\.\d+)?\s*(?:pt|em|ex|mm|cm|in|bp|px)\s*\]", "", s)
    s = _MATH_DISPLAY.sub(lambda m: "$$"+_fix_overescape_in_math(m.group(1))+"$$", s)
    s = _MATH_INLINE.sub (lambda m: r"\("+_fix_overescape_in_math(m.group(1))+r"\)", s)
    return s

def extract_json_block(text: str) -> str:
    if not text: return ""
    start = text.find("{"); end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        cand = text[start:end+1]
        try: json.loads(cand); return cand
        except: pass
    return ""

def responses_result_to_text(resp) -> str:
    try: return resp.output[0].content[0].text
    except Exception: pass
    try: return resp.output_text or ""
    except Exception: return ""

def get_payload_json() -> dict:
    data = request.get_json(silent=True)
    if isinstance(data, dict): return data
    raw = request.data or b""
    if raw:
        try: return json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception: pass
    return {}

# ── CORS / health / error handling ────────────────────────────────────────────
@app.after_request
def add_cors_headers(resp):
    resp.headers.setdefault("Access-Control-Allow-Origin", "*" if FRONTEND_ORIGIN == "*" else FRONTEND_ORIGIN)
    resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization")
    resp.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return resp

@app.route("/chat", methods=["OPTIONS"])
def chat_options(): return ("", 204)

@app.get("/ping")
def ping(): return jsonify({"status": "ok"})

@app.errorhandler(Exception)
def handle_any_error(e):
    # Ensure 500s are returned as JSON, not HTML
    tb = traceback.format_exc()
    print(tb, flush=True)
    payload = {"error": f"{type(e).__name__}: {e}"}
    if DEBUG_ERRORS: payload["trace"] = tb
    return jsonify(payload), 500

# ── Tutor helper ──────────────────────────────────────────────────────────────
def get_full_answer(question: str) -> str:
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": LATEX_SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    return sanitize_latex(resp.output_text or "")

# ── LLM JSON shim (handles SDK differences) ───────────────────────────────────
def call_llm_json(system_prompt: str, user_messages: list, model: str) -> Tuple[Optional[dict], Optional[str]]:
    # 1) Responses with response_format
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "system", "content": system_prompt}] + user_messages,
            response_format={"type": "json_object"},
        )
        text = responses_result_to_text(resp)
        try: return json.loads(text), text
        except: 
            block = extract_json_block(text)
            if block: return json.loads(block), text
    except TypeError:
        pass
    except Exception as e:
        print("Responses+json_object error:", e, flush=True)

    # 2) Responses without response_format
    try:
        strengthened = user_messages + [{"role": "user", "content": "Output ONLY JSON. Begin with '{' and end with '}'."}]
        resp = client.responses.create(
            model=model,
            input=[{"role": "system", "content": system_prompt}] + strengthened,
        )
        text = responses_result_to_text(resp)
        block = extract_json_block(text)
        if block: return json.loads(block), text
    except Exception as e:
        print("Responses (no json mode) error:", e, flush=True)

    # 3) Chat completions fallback
    try:
        msgs = [{"role": "system", "content": system_prompt}] + user_messages + [
            {"role": "user", "content": "Return ONLY JSON. Start with '{' and end with '}'."}
        ]
        try:
            resp = client.chat.completions.create(model=model, messages=msgs, response_format={"type": "json_object"})
            text = resp.choices[0].message.content or ""
            return json.loads(text), text
        except TypeError:
            resp = client.chat.completions.create(model=model, messages=msgs)
            text = resp.choices[0].message.content or ""
            block = extract_json_block(text)
            if block: return json.loads(block), text
    except Exception as e:
        print("ChatCompletions error:", e, flush=True)

    return None, None

# ── Analytics handler ─────────────────────────────────────────────────────────
def handle_analyze_adaptive_quiz(payload: dict):
    """
    Payload:
      { "mode":"analyze_adaptive_quiz",
        "schema":[...], "csv":"header\\nrows...", "run_label":"...", "dryrun": bool }
    """
    schema = payload.get("schema") or ["userid","username","quizname","difficultysum","standarderror","measure","timetaken"]
    csv_text = (payload.get("csv") or "").strip()
    run_label = payload.get("run_label") or f"manual_{time.strftime('%Y-%m-%d')}"
    user_message = payload.get("message") or "Analyze the CSV and return strict JSON."
    dryrun = bool(payload.get("dryrun"))

    if not csv_text:
        return jsonify({"error": "Missing 'csv' content"}), 400

    # DRY-RUN path to verify end-to-end without OpenAI
    if dryrun or not OPENAI_API_KEY:
        # Build a tiny fake output using just the header + first data row (if present)
        lines = [ln for ln in csv_text.splitlines() if ln.strip()]
        hdr = lines[0].split(",") if lines else []
        items = []
        for row in lines[1:3]:  # at most 2 demo rows
            cols = row.split(",")
            rec = dict(zip(hdr, cols))
            uid = int(rec.get("userid", "0") or 0)
            items.append({
                "userid": uid,
                "risk_score": 50.0,
                "confidence": 0.3,
                "drivers": ["dryrun mode"],
                "student_msg": "This is a dry-run preview.",
                "teacher_msg": "Dry-run: verify data flow.",
                "features": rec
            })
        return jsonify({"run_label": run_label, "items": items})

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

    json_obj, _ = call_llm_json(
        system_prompt=ANALYTICS_SYSTEM,
        user_messages=[
            {"role": "user", "content": user_message},
            {"role": "user", "content": prompt},
        ],
        model=MODEL,
    )

    if not isinstance(json_obj, dict):
        return jsonify({"error": "LLM_JSON_PARSE_FAILED"}), 200

    json_obj.setdefault("run_label", run_label)
    if "items" not in json_obj or not isinstance(json_obj["items"], list):
        json_obj["items"] = []

    return jsonify(json_obj)

# ── /chat with robust mode switching ──────────────────────────────────────────
@app.post("/chat")
def chat():
    payload = get_payload_json()
    mode = (payload.get("mode") or request.args.get("mode") or "chat").strip().lower()

    # If it looks like analytics (csv/schema present), force analytics mode
    if mode == "chat" and (("csv" in payload) or ("schema" in payload)):
        mode = "analyze_adaptive_quiz"

    if mode == "analyze_adaptive_quiz":
        return handle_analyze_adaptive_quiz(payload)

    # Tutor path
    question = (payload.get("message") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'message'"}), 400

    if not STREAMING_DEFAULT:
        cached = answer_cache.get(question)
        if cached: return jsonify({"reply": cached})
        ans = get_full_answer(question)
        answer_cache.put(question, ans)
        return jsonify({"reply": ans})

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
                yield sanitize_latex("".join(parts))
        except Exception as e:
            yield f"\n[Stream error: {type(e).__name__}: {e}]"
        finally:
            text = "".join(parts).strip()
            if text: answer_cache.put(question, sanitize_latex(text))

    return Response(stream_with_context(generate()), mimetype="text/plain; charset=utf-8")

# ── Keep-alive (optional) ─────────────────────────────────────────────────────
def keep_alive():
    url = os.getenv("SELF_PING_URL")
    if not url: return
    try:
        import requests
    except Exception:
        return
    while True:
        try: requests.get(url + "/ping", timeout=5)
        except Exception: pass
        time.sleep(600)

if os.getenv("SELF_PING_URL"):
    threading.Thread(target=keep_alive, daemon=True).start()

# ── Local dev runner ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
