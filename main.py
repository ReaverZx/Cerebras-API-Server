from flask import Flask, request, jsonify
from cerebras.cloud.sdk import Cerebras
import os

app = Flask(__name__)

# Initialize the Cerebras client
client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.json
    user_message = data.get("message", "")

    # Create the stream (just like your script)
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": user_message}],
        model="gpt-oss-120b",
        stream=True,
        max_completion_tokens=65536,
        temperature=1,
        top_p=1,
        reasoning_effort="medium"
    )

    # Gather output
    response_text = ""
    for chunk in stream:
        response_text += chunk.choices[0].delta.content or ""

    return jsonify({"response": response_text})

if __name__ == "__main__":
    # Listen on port 10000 (Render will set $PORT automatically)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
