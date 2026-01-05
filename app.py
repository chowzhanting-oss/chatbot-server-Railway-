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
# System prompt: concise + clean LaTeX rules
# ──────────────────────────────────────────────────────────────────────────────
LATEX_SYSTEM = (
    "You are a patient electronics tutor for Integrated Electronics "
    "(CMOS, MOSFETs, amplifiers, threshold voltage, etc.). "
    "Be concise and cost-efficient. Use the minimum words needed to fully answer the question "
    "without omitting crucial information. "
    "Do not add background unless required. "
    "Provide derivations only if explicitly asked. "
    "Write in short, clean paragraphs. Do not use bullet points or hyphenated lists. "

    "Math formatting rules (very important): "
    "Whenever you write an equation or formula, write it in LaTeX. "
    "Use $$ ... $$ for display equations (centered, on their own line) and \\( ... \\) for inline math. "
    "Use TeX commands, not Unicode symbols (e.g., write \\mu, \\cdot, \\frac{...}{...}). "
    "Use subscripts/superscripts properly: x_{...}, x^{...}. "
    "Use C_{ox}, V_{GS}, V_{DS}, V_T, I_{D,\\mathrm{lin}} (or I_{D,lin}) as appropriate. "

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
# LaTeX sanitizers + Unicode-to-LaTeX normalization
# ──────────────────────────────────────────────────────────────────────────────
def normalize_unicode_math(s: str) -> str:
    """
    Convert common Unicode math glyphs into LaTeX-friendly forms.
    This helps MathJax render consistently like your reference image.
    """
    if not s:
        return s

    # Basic symbol normalization
    replacements = {
        "μ": r"\mu",
        "·": r"\cdot",
        "×": r"\times",
        "−": "-",      # unicode minus
        "–": "-",      # en dash
        "—": "-",      # em dash
        "π": r"\pi",
        "Ω": r"\Omega",
        "α": r"\alpha",
        "β": r"\beta",
        "γ": r"\gamma",
        "Δ": r"\Delta",
        "θ": r"\theta",
        "λ": r"\lambda",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    # Superscript digits → ^{digit}
    s = s.replace("²", r"^{2}")
    s = s.replace("³", r"^{3}")
    s = s.replace("⁴", r"^{4}")
    s = s.replace("⁵", r"^{5}")
    s = s.replace("⁶", r"^{6}")
    s = s.replace("⁷", r"^{7}")
    s = s.replace("⁸", r"^{8}")
    s = s.replace("⁹", r"^{9}")
    s = s.replace("⁰", r"^{0}")

    return s

def _collapse_command_backslashes(s: str) -> str:
    # Convert \\mu, \\frac, \\left, \\[, \\] → \mu, \frac, \left, \[, \]
    # but DO NOT destroy line breaks like "\\ " or "\\\n".
    s = re.sub(r"\\\\(?=[A-Za-z\[\]])", r"\\", s)
    s = re.sub(r"\\{3,}", r"\\", s)
    return s

def _fix_overescape_in_math(math: str) -> str:
    """
    Inside a math block only:
    - fix \left\[ -> \left[ and \right\] -> \right]
    - remove stray backslashes before operators/brackets: \= \( \) \[ \] \+ \- \* / ^ _
    """
    math = math.replace(r"\left\[", r"\left[").replace(r"\right\]", r"\right]")
    math = re.sub(r"\\([=\(\)\[\]\+\-\*/\^_])", r"\1", math)
    return math

_MATH_DISPLAY = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_MATH_INLINE  = re.compile(r"\\\((.*?)\\\)", re.DOTALL)

def sanitize_latex(s: str) -> str:
    """
    Make model output friendlier to MathJax:
      • normalize unicode math glyphs into TeX,
      • collapse \\ before commands/brackets, preserving real line breaks,
      • strip [6pt]/[8pt]/[12mm]/[0.5em]/etc.,
      • fix over-escaped punctuation/brackets inside $$...$$ and \(...\).
    """
    # 0) Normalize unicode math symbols (μ, ·, ², etc.) into TeX
    s = normalize_unicode_math(s)

    # 1) Clean backslashes for commands (keep \\ line breaks intact)
    s = _collapse_command_backslashes(s)

    # 2) Remove TeX spacing hints like [6pt], [ 0.5 em ], [12mm], [8px], etc.
    s = re.sub(r"\[\s*\d+(?:\.\d+)?\s*(?:pt|em|ex|mm|cm|in|bp|px)\s*\]", "", s)

    # 3) Clean inside math blocks
    def _fix_display(m): return "$$" + _fix_overescape_in_math(m.group(1)) + "$$"
    def _fix_inline(m):  return r"\(" + _fix_overescape_in_math(m.group(1)) + r"\)"
    s = _MATH_DISPLAY.sub(_fix_display, s)
    s = _MATH_INLINE.sub(_fix_inline, s)

    # 4) Encourage display equations to be on their own line (MathJax renders better)
    s = re.sub(r"\s*\$\$(.*?)\$\$\s*", lambda m: "\n\n$$" + m.group(1).strip() + "$$\n\n", s, flags=re.DOTALL)

    return s

# ──────────────────────────────────────────────────────────────────────────────
# Text formatting: remove bullets + split dense paragraphs + justify safely
# ──────────────────────────────────────────────────────────────────────────────
JUSTIFY_WIDTH = 80

def _justify_paragraph(words, width):
    if len(words) == 1:
        return words[0]

    total_chars = sum(len(w) for w in words)
    gaps = len(words) - 1

    if total_chars >= width or gaps <= 0:
        return " ".join(words)

    spaces_needed = width - total_chars
    space, extra = divmod(spaces_needed, gaps)

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
        if (
            not stripped
            or "$" in stripped
            or r"\(" in stripped
            or r"\)" in stripped
        ):
            output.append(line)
            continue

        words = stripped.split()
        if not words:
            output.append(stripped)
            continue

        total_chars = sum(len(w) for w in words)
        if total_chars < JUSTIFY_WIDTH and (total_chars + (len(words) - 1)) >= JUSTIFY_WIDTH:
            output.append(_justify_paragraph(words, JUSTIFY_WIDTH))
        else:
            output.append(" ".join(words))

    return "\n".join(output)

def split_long_paragraphs(text: str, max_len: int = 300) -> str:
    """
    Split overly dense paragraphs into multiple paragraphs at sentence boundaries
    for readability, while avoiding interference with MathJax/LaTeX.
    """
    paragraphs = text.split("\n\n")
    new_paragraphs = []

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        # Don't touch math-heavy paragraphs
        if len(p) <= max_len or "$" in p or r"\(" in p or r"\)" in p:
            new_paragraphs.append(p)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", p)
        chunk = ""
        for s in sentences:
            if not s:
                continue
            if not chunk:
                chunk = s
                continue
            if len(chunk) + 1 + len(s) <= max_len:
                chunk += " " + s
            else:
                new_paragraphs.append(chunk)
                chunk = s
        if chunk:
            new_paragraphs.append(chunk)

    return "\n\n".join(new_paragraphs)

def format_reply(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s

    # Remove bullet / hyphen prefixes
    s = re.sub(r"(?m)^\s*(?:[-•*–—]\s+)+", "", s)

    # Normalize blank lines
    s = re.sub(r"\n{3,}", "\n\n", s)

    # Split dense paragraphs into multiple paragraphs
    s = split_long_paragraphs(s)

    # Justify safely (note: skip math lines)
    s = justify_text(s)

    return s.strip()

# ──────────────────────────────────────────────────────────────────────────────
# CORS
# ──────────────────────────────────────────────────────────────────────────────
@app.after_request
def add_cors_headers(resp):
    resp.headers.setdefault(
        "Access-Control-Allow-Origin",
        "*" if FRONTEND_ORIGIN == "*" else FRONTEND_ORIGIN
    )
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

    return Response(
        stream_with_context(generate()),
        mimetype="text/plain; charset=utf-8"
    )

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
