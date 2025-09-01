import os
from flask import Flask, request, jsonify
from cerebras.cloud.sdk import Cerebras

app = Flask(__name__)

client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)

    message = data.get("message")
    model = data.get("model", "gpt-oss-7b")  # default if none given

    if not message:
        return jsonify({"error": "Missing 'message' field"}), 400

    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": message}],
            model=model,
            max_completion_tokens=512
        )

        reply = None
        if completion.choices:
            choice = completion.choices[0]
            if hasattr(choice.message, "content"):
                reply = choice.message.content
            elif isinstance(choice.message, dict) and "content" in choice.message:
                reply = choice.message["content"]
            elif hasattr(choice, "text"):
                reply = choice.text

        if not reply:
            return jsonify({"error": "No content in response", "raw": str(completion)}), 500

        return jsonify({"response": reply, "model_used": model})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
