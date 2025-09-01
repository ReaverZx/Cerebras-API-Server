import os
from flask import Flask, request, jsonify
from cerebras.cloud.sdk import Cerebras

app = Flask(__name__)

client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY")
)

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)

    # Expect both "message" and "model"
    message = data.get("message", "")
    model = data.get("model", "gpt-oss-7b")

    if not message:
        return jsonify({"error": "Missing 'message' field"}), 400

    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": message}],
            model=model,
            max_completion_tokens=1024
        )

        reply = completion.choices[0].message["content"]
        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
