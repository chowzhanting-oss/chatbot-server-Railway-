import os
import threading
import time
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Enable/disable streaming via environment (default = on)
STREAMING_DEFAULT = os.getenv("STREAMING", "on").lower() == "on"

# Cache recent answers in memory
answer_cache = {}

LATEX_SYSTEM = (
    "You are a patient electronics tutor for Integrated Electronics. "
    "Default behavior: respond briefly and clearly using short bullet points or short paragraphs. "
    "Always format mathematical expressions using LaTeX between double dollar signs ($$ ... $$). "
    "Example: $$ I_D = \\mu_n C_{ox} \\frac{W}{L}[(V_{GS}-V_{TH})V_{DS}-\\frac{V_{DS}^2}{2}] $$. "
    "Only expand with detailed derivations if the user explicitly asks to 'explain more' or 'show derivation'. "
    "If the question is off-topic, reply exactly: "
    "Sorry I cannot help you with that, I can only answer questions about Integrated Electronics."
)

def non_streaming_answer(question: str) -> str:
    """Plain (non-streaming) call; returns full text."""
    resp = client.responses.create(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": LATEX_SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    return resp.output_text or ""

@app.get("/ping")
def ping():
    return jsonify({"status": "ok"})

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    if not question:
        return jsonify({"error": "No message received"}), 400

    # Return cached answer if available
    if question in answer_cache:
        return jsonify({"reply": answer_cache[question], "cached": True})

    # Non-streaming path
    if not STREAMING_DEFAULT:
        answer = non_streaming_answer(question)
        answer_cache[question] = answer
        return jsonify({"reply": answer})

    # Streaming path
    def generate():
        collected = []
        try:
            with client.responses.stream(
                model="gpt-5-mini",
                input=[
                    {"role": "system", "content": LATEX_SYSTEM},
                    {"role": "user", "content": question},
                ],
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        chunk = event.delta
                        collected.append(chunk)
                        yield chunk
                stream.close()
            full = "".join(collected)
            answer_cache[question] = full
        except Exception as e:
            yield f"[Error: {e}]"

    return Response(generate(), mimetype="text/plain")

def keep_alive():
    while True:
        try:
            with app.test_client() as c:
                c.get("/ping")
        except:
            pass
        time.sleep(600)  # every 10 min

threading.Thread(target=keep_alive, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
