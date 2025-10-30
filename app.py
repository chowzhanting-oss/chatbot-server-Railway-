# app.py
import os, re, time, json, threading, traceback
from collections import OrderedDict
from typing import Optional, Tuple

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
STREAMING_DEFAULT = os.getenv("STREAMING", "on").strip().lower() == "on"
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
DEBUG_ERRORS = os.getenv("DEBUG_ERRORS", "on").strip().lower() == "on"

app = Flask(__name__)
if FRONTEND_ORIGIN == "*":
    CORS(app)
else:
    CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})

client = OpenAI(api_key=OPENAI_API_KEY)

# ── System Prompts ────────────────────────────────────────────────────────────
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

# ── LRU Cache ────────────────────────────────────────────────────────────────
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

# ── LaTeX Sanitizers ─────────────────────────────────────────────────────────
def sanitize_latex(s: str) -> str:
    if not s: return ""
    # fix escaped characters from JSON/text issues
    s = s.replace("\\$", "$").replace("\\\\\\", "\\\\")
    s = re.sub(r"\\\\(?=[A-Za-z\[\]])", r"\\", s)
    s = re.sub(r"\\{3,}", r"\\", s)
    s = re.sub(r"\\([=\(\)\[\]\+\-\*/\^_])", r"\1", s)
    s = s.replace(r"\left\[", r"\left[").replace(r"\right\]", r"\right]")
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

# ── Error Handling / Ping ────────────────────────────────────────────────────
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
    tb = traceback.format_exc()
    print(tb, flush=True)
    payload = {"error": f"{type(e).__name__}: {e}"}
    if DEBUG_ERRORS: payload["trace"] = tb
    return jsonify(payload), 500

# ── Tutor Mode ───────────────────────────────────────────────────────────────
def get_full_answer(question: str) -> str:
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": LATEX_SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    return sanitize_latex(resp.output_text or "")

@app.post("/chat")
def chat():
    payload = get_payload_json()
    question = (payload.get("message") or "").strip()

    if not question:
        return jsonify({"error": "Missing 'message'"}), 400

    # Check cache
    cached = answer_cache.get(question)
    if cached: return jsonify({"reply": cached})

    # Non-streaming answer path
    if not STREAMING_DEFAULT:
        ans = get_full_answer(question)
        answer_cache.put(question, ans)
        return jsonify({"reply": ans})

    # Streaming mode
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
                        yield sanitize_latex(event.delta or "")
        except Exception as e:
            yield f"\n[Stream error: {type(e).__name__}: {e}]"
        finally:
            text = "".join(parts).strip()
            if text:
                answer_cache.put(question, sanitize_latex(text))

    return Response(stream_with_context(generate()), mimetype="text/plain; charset=utf-8")

# ── JSON Mode Helper ─────────────────────────────────────────────────────────
def call_llm_json(system_prompt: str, user_messages: list, model: str) -> Tuple[Optional[dict], Optional[str]]:
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
    except Exception as e:
        print("JSON call error:", e, flush=True)
    return None, None

# ── Analytics Handler ────────────────────────────────────────────────────────
def handle_analyze_adaptive_quiz(payload: dict):
    schema = payload.get("schema") or ["userid","username","quizname","difficultysum","standarderror","measure","timetaken"]
    csv_text = (payload.get("csv") or "").strip()
    run_label = payload.get("run_label") or f"manual_{time.strftime('%Y-%m-%d')}"
    user_message = payload.get("message") or "Analyze the CSV and return strict JSON."
    dryrun = bool(payload.get("dryrun"))

    if not csv_text:
        return jsonify({"error": "Missing 'csv' content"}), 400

    # Dry-run preview
    if dryrun or not OPENAI_API_KEY:
        lines = [ln for ln in csv_text.splitlines() if ln.strip()]
        hdr = lines[0].split(",") if lines else []
        items = []
        for row in lines[1:3]:
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
Return a SINGLE valid JSON object with this shape:

{{
  "run_label": string,
  "items": [
    {{
      "userid": int,
      "risk_score": number,
      "confidence": number,
      "drivers": [string],
      "student_msg": string,
      "teacher_msg": string,
      "features": object
    }}
  ]
}}
Rules:
- Output ONLY JSON.
- If data is insufficient, keep risk_score ~50 and confidence low.
- Aggregate rows by userid.
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

# ── Dedicated Analytics Endpoint ─────────────────────────────────────────────
@app.post("/analyze")
def analyze():
    payload = get_payload_json()
    return handle_analyze_adaptive_quiz(payload)

# ── Keep Alive ───────────────────────────────────────────────────────────────
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

# ── Local Run ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
