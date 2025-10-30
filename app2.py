# app2.py  -- analytics-only Flask server
import os, time, json, traceback, re
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
DEBUG_ERRORS = os.getenv("DEBUG_ERRORS", "on").strip().lower() == "on"

app = Flask(__name__)
if FRONTEND_ORIGIN == "*":
    CORS(app)
else:
    CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})

client = OpenAI(api_key=OPENAI_API_KEY)

# ── System prompt ─────────────────────────────────────────────────────────
ANALYTICS_SYSTEM = (
    "You are a strict learning analytics assistant. "
    "Return a single valid JSON object according to the requested schema. "
    "Output ONLY JSON, no prose."
)

def extract_json_block(text: str) -> str:
    if not text: return ""
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        cand = text[start:end+1]
        try: json.loads(cand); return cand
        except: pass
    return ""

def call_llm_json(system_prompt: str, user_messages: list):
    try:
        resp = client.responses.create(
            model=MODEL,
            input=[{"role": "system", "content": system_prompt}] + user_messages,
            response_format={"type": "json_object"},
        )
        text = resp.output[0].content[0].text
        try:
            return json.loads(text)
        except:
            block = extract_json_block(text)
            if block: return json.loads(block)
    except Exception as e:
        print("LLM JSON error:", e, flush=True)
    return {"error": "LLM_JSON_PARSE_FAILED"}

# ── Analytics endpoint ────────────────────────────────────────────────────
@app.post("/analyze")
def analyze():
    payload = request.get_json(silent=True) or {}
    schema = payload.get("schema") or [
        "userid","username","quizname","difficultysum",
        "standarderror","measure","timetaken"
    ]
    csv_text = (payload.get("csv") or "").strip()
    run_label = payload.get("run_label") or f"manual_{time.strftime('%Y-%m-%d')}"
    dryrun = bool(payload.get("dryrun"))

    if not csv_text:
        return jsonify({"error": "Missing 'csv' content"}), 400

    # Dry-run demo
    if dryrun or not OPENAI_API_KEY:
        lines=[ln for ln in csv_text.splitlines() if ln.strip()]
        hdr=lines[0].split(",") if lines else []
        items=[]
        for row in lines[1:3]:
            cols=row.split(","); rec=dict(zip(hdr,cols))
            uid=int(rec.get("userid","0") or 0)
            items.append({
                "userid": uid,
                "risk_score": 50.0,
                "confidence": 0.3,
                "drivers": ["dryrun mode"],
                "student_msg": "Dry-run preview.",
                "teacher_msg": "Verify data flow.",
                "features": rec
            })
        return jsonify({"run_label": run_label, "items": items})

    schema_str = ", ".join(schema)
    prompt = f"""
You will be given a CSV with columns: {schema_str}.
Return this JSON shape:

{{
  "run_label": "{run_label}",
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
Output ONLY JSON.
If uncertain, use risk_score≈50 and confidence≈0.3.
Aggregate rows by userid.
CSV data:
{csv_text}
""".strip()

    json_obj = call_llm_json(
        ANALYTICS_SYSTEM,
        [{"role": "user", "content": prompt}],
    )
    return jsonify(json_obj)

@app.get("/ping")
def ping(): return jsonify({"status": "ok"})

@app.errorhandler(Exception)
def handle_err(e):
    tb = traceback.format_exc()
    print(tb, flush=True)
    msg = {"error": f"{type(e).__name__}: {e}"}
    if DEBUG_ERRORS: msg["trace"] = tb
    return jsonify(msg), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))  # run on different port
    app.run(host="0.0.0.0", port=port, debug=False)
