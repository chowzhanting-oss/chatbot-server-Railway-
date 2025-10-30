import os, re, time, json, threading, traceback
from collections import OrderedDict
from typing import Optional, Tuple
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────── 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # safer default
STREAMING_DEFAULT = os.getenv("STREAMING", "on").strip().lower() == "on"
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
DEBUG_ERRORS = os.getenv("DEBUG_ERRORS", "on").strip().lower() == "on"  # show trace in JSON errors

app = Flask(__name__)
if FRONTEND_ORIGIN == "*":
    CORS(app)
else:
    CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})

client = OpenAI(api_key=OPENAI_API_KEY)

# ── Chatbot Section ────────────────────────────────────────────────────────────
LATEX_SYSTEM = (
    "You are a patient electronics tutor for Integrated Electronics (CMOS, MOSFETs, amplifiers, threshold voltage, etc.). "
    "Default: respond briefly and clearly. Use LaTeX with $$...$$ for display math and \\(...\\) for inline math. "
    "If the user greets you casually (hi, hello, hey, etc.), reply with a friendly welcome such as 'Hello! How can I help you today?'. "
    "Only when a request is clearly outside electronics should you reply exactly: Sorry I cannot help you with that, I can only answer questions about Integrated Electronics."
)

def get_full_answer(question: str) -> str:
    """ Function to get the full answer from GPT in LaTeX format """
    resp = client.responses.create(
        model=MODEL,
        input=[{"role": "system", "content": LATEX_SYSTEM},
               {"role": "user", "content": question}]
    )
    return resp.output_text or ""  # Return the LaTeX-formatted response

@app.post("/chat")
def chat():
    """ Chatbot route to handle incoming messages """
    payload = get_payload_json()
    question = (payload.get("message") or "").strip()
    
    if not question:
        return jsonify({"error": "Missing 'message'"}), 400
    
    # Get the LaTeX-formatted response from GPT
    answer = get_full_answer(question)
    
    return jsonify({"reply": answer})


# ── Analytics Section ──────────────────────────────────────────────────────────
ANALYTICS_SYSTEM = (
    "You are a strict learning analytics assistant. "
    "Return a single valid JSON object matching the requested schema. "
    "No prose outside JSON. Be conservative with risk when confidence is low."
)

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

def extract_json_block(text: str) -> str:
    if not text: return ""
    start = text.find("{"); end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        cand = text[start:end+1]
        try: json.loads(cand); return cand
        except: pass
    return ""

def handle_analyze_adaptive_quiz(payload: dict):
    """
    Payload:
      { "mode":"analyze_adaptive_quiz",
        "schema":[...], "csv":"header\\nrows...", "run_label":"...", "dryrun": bool }
    """
    schema = payload.get("schema") or ["userid", "username", "quizname", "difficultysum", "standarderror", "measure", "timetaken"]
    csv_text = (payload.get("csv") or "").strip()
    run_label = payload.get("run_label") or f"manual_{time.strftime('%Y-%m-%d')}"
    user_message = payload.get("message") or "Analyze the CSV and return strict JSON."
    dryrun = bool(payload.get("dryrun"))

    if not csv_text:
        return jsonify({"error": "Missing 'csv' content"}), 400

    # DRY-RUN path to verify end-to-end without OpenAI
    if dryrun or not OPENAI_API_KEY:
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

    CSV data:
    {csv_text}
    """.strip()

    json_obj, _ = call_llm_json(
        system_prompt=ANALYTICS_SYSTEM,
        user_messages=[{"role": "user", "content": user_message}, {"role": "user", "content": prompt}],
        model=MODEL,
    )

    if not isinstance(json_obj, dict):
        return jsonify({"error": "LLM_JSON_PARSE_FAILED"}), 200

    json_obj.setdefault("run_label", run_label)
    if "items" not in json_obj or not isinstance(json_obj["items"], list):
        json_obj["items"] = []

    return jsonify(json_obj)

@app.route("/analyze", methods=["POST"])
def analyze():
    payload = get_payload_json()

    if payload.get("mode") == "analyze_adaptive_quiz":
        return handle_analyze_adaptive_quiz(payload)

    return jsonify({"error": "Invalid mode"}), 400


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
