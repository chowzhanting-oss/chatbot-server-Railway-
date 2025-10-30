# app.py
import os, re, time, threading, json, traceback
from collections import OrderedDict
from typing import Optional, Tuple
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Environment / Config
# ──────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
STREAMING_DEFAULT = os.getenv("STREAMING", "on").strip().lower() == "on"
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

app = Flask(__name__)
if FRONTEND_ORIGIN == "*":
    CORS(app)
else:
    CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})

client = OpenAI(api_key=OPENAI_API_KEY)

# ──────────────────────────────────────────────────────────────────────────────
# System Prompts
# ──────────────────────────────────────────────────────────────────────────────
LATEX_SYSTEM = (
    "You are a patient electronics tutor for Integrated Electronics "
    "(CMOS, MOSFETs, amplifiers, threshold voltage, etc.). "
    "Default: respond briefly and clearly. "
    "Use LaTeX for math: one display block with $$...$$ for multi-line equations "
    "and \\(...\\) for inline math. "
    "Do NOT escape punctuation/brackets inside math (write = ( ) [ ] ^ _ plainly). "
    "Avoid layout directives like [6pt], [8pt], etc. "
    "Example: $$ I_D = \\mu_n C_{ox}\\frac{W}{L}[(V_{GS}-V_T)V_{DS}-\\frac{V_{DS}^2}{2}] $$. "
    "If the question is off-topic, reply exactly: "
    "Sorry I cannot help you with that, I can only answer questions about Integrated Electronics."
)

ANALYTICS_SYSTEM = (
    "You are a strict learning analytics assistant. "
    "Return only valid JSON according to the requested schema. "
    "No prose, no Markdown, no commentary."
)

# ──────────────────────────────────────────────────────────────────────────────
# Utilities: cache + LaTeX cleaner
# ──────────────────────────────────────────────────────────────────────────────
class LRU(OrderedDict):
    def __init__(self, maxsize=64): super().__init__(); self.maxsize=maxsize
    def get(self, k): 
        if k in self:
            v=super().pop(k); super().__setitem__(k,v); return v
        return None
    def put(self, k,v):
        if k in self: super().pop(k)
        elif len(self)>=self.maxsize: self.popitem(last=False)
        super().__setitem__(k,v)
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
    if not s: return ""
    s = _collapse_command_backslashes(s)
    s = re.sub(r"\[\s*\d+(?:\.\d+)?\s*(?:pt|em|mm|cm|in|bp|px)\s*\]", "", s)
    s = _MATH_DISPLAY.sub(lambda m: "$$"+_fix_overescape_in_math(m.group(1))+"$$", s)
    s = _MATH_INLINE.sub(lambda m: r"\("+_fix_overescape_in_math(m.group(1))+r"\)", s)
    return s

def extract_json_block(text: str) -> str:
    if not text: return ""
    start, end = text.find("{"), text.rfind("}")
    if start!=-1 and end!=-1 and end>start:
        cand=text[start:end+1]
        try: json.loads(cand); return cand
        except: return ""
    return ""

@app.after_request
def add_cors_headers(resp):
    resp.headers.setdefault("Access-Control-Allow-Origin","*" if FRONTEND_ORIGIN=="*" else FRONTEND_ORIGIN)
    resp.headers.setdefault("Access-Control-Allow-Headers","Content-Type, Authorization")
    resp.headers.setdefault("Access-Control-Allow-Methods","GET, POST, OPTIONS")
    return resp

@app.get("/ping")
def ping(): return jsonify({"status":"ok"})

# ──────────────────────────────────────────────────────────────────────────────
# Tutor: /chat
# ──────────────────────────────────────────────────────────────────────────────
def get_full_answer(question: str) -> str:
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role":"system","content":LATEX_SYSTEM},
            {"role":"user","content":question},
        ]
    )
    return sanitize_latex(resp.output_text or "")

@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("message") or "").strip()
    if not question:
        return jsonify({"error":"Missing 'message'"}),400

    cached = answer_cache.get(question)
    if cached:
        return jsonify({"reply":cached})

    if not STREAMING_DEFAULT:
        ans = get_full_answer(question)
        answer_cache.put(question, ans)
        return jsonify({"reply":ans})

    def generate():
        parts=[]
        try:
            with client.responses.stream(
                model=MODEL,
                input=[
                    {"role":"system","content":LATEX_SYSTEM},
                    {"role":"user","content":question},
                ],
            ) as stream:
                for event in stream:
                    if event.type=="response.output_text.delta":
                        parts.append(event.delta or "")
                yield sanitize_latex("".join(parts))
        except Exception as e:
            yield f"\n[Stream error: {type(e).__name__}: {e}]"
        finally:
            text="".join(parts).strip()
            if text: answer_cache.put(question, sanitize_latex(text))

    return Response(stream_with_context(generate()), mimetype="text/plain; charset=utf-8")

# ──────────────────────────────────────────────────────────────────────────────
# Analytics: /analyze
# ──────────────────────────────────────────────────────────────────────────────
def call_llm_json(system_prompt: str, user_messages: list) -> Tuple[Optional[dict], Optional[str]]:
    try:
        resp = client.responses.create(
            model=MODEL,
            input=[{"role":"system","content":system_prompt}] + user_messages,
            response_format={"type":"json_object"},
        )
        text = resp.output[0].content[0].text
        try:
            return json.loads(text), text
        except:
            block = extract_json_block(text)
            if block: return json.loads(block), text
    except Exception as e:
        print("JSON call error:", e, flush=True)
    return None, None

@app.post("/analyze")
def analyze():
    payload = request.get_json(silent=True) or {}
    schema = payload.get("schema") or ["userid","username","quizname","difficultysum","standarderror","measure","timetaken"]
    csv_text = (payload.get("csv") or "").strip()
    run_label = payload.get("run_label") or f"manual_{time.strftime('%Y-%m-%d')}"
    dryrun = bool(payload.get("dryrun"))
    if not csv_text:
        return jsonify({"error":"Missing 'csv' content"}),400

    if dryrun or not OPENAI_API_KEY:
        lines=[ln for ln in csv_text.splitlines() if ln.strip()]
        hdr=lines[0].split(",") if lines else []
        items=[]
        for row in lines[1:3]:
            cols=row.split(",")
            rec=dict(zip(hdr,cols))
            uid=int(rec.get("userid","0") or 0)
            items.append({
                "userid":uid,
                "risk_score":50.0,
                "confidence":0.3,
                "drivers":["dryrun mode"],
                "student_msg":"Dry-run: verifying data flow.",
                "teacher_msg":"Dry-run: verifying data flow.",
                "features":rec
            })
        return jsonify({"run_label":run_label,"items":items})

    prompt=f"""
You will be given CSV data with columns: {', '.join(schema)}.
Return a JSON object of this form:

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
If data is insufficient, keep risk_score ≈50 and confidence low.
CSV data:
{csv_text}
""".strip()

    json_obj,_ = call_llm_json(
        ANALYTICS_SYSTEM,
        [{"role":"user","content":prompt}],
    )
    if not isinstance(json_obj, dict):
        return jsonify({"error":"LLM_JSON_PARSE_FAILED"}),200
    return jsonify(json_obj)

# ──────────────────────────────────────────────────────────────────────────────
# Keep-alive
# ──────────────────────────────────────────────────────────────────────────────
def keep_alive():
    url=os.getenv("SELF_PING_URL")
    if not url: return
    try: import requests
    except Exception: return
    while True:
        try: requests.get(url+"/ping",timeout=5)
        except Exception: pass
        time.sleep(600)
if os.getenv("SELF_PING_URL"):
    threading.Thread(target=keep_alive,daemon=True).start()

if __name__=="__main__":
    port=int(os.environ.get("PORT",8080))
    app.run(host="0.0.0.0",port=port,debug=False)
