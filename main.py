from flask import Flask, request, jsonify
from flask_cors import CORS
from cerebras.cloud.sdk import Cerebras
import os
import traceback

app = Flask(__name__)
# Allow cross-origin requests (useful for HTML5 exports). Adjust origins if needed.
CORS(app)

# Initialize client with API key from env
client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

@app.route("/api/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        # Accept either a plain message string or prebuilt messages array
        model = data.get("model", "gpt-oss-120b")
        # `message` expected to be a single string; `messages` can be an array of role/content objects.
        if "messages" in data and isinstance(data["messages"], list):
            messages = data["messages"]
        else:
            user_message = data.get("message", "")
            if not user_message:
                return jsonify({"error": "No 'message' provided"}), 400
            messages = [{"role": "user", "content": user_message}]

        # Call Cerebras (streaming, but we collect into one string and return it)
        stream = client.chat.completions.create(
            messages=messages,
            model=model,
            stream=True,
            max_completion_tokens=65536,
            temperature=data.get("temperature", 1),
            top_p=data.get("top_p", 1),
            reasoning_effort=data.get("reasoning_effort", "medium"),
        )

        response_text = ""
        # Iterate stream and concatenate content pieces
        for chunk in stream:
            # Some chunks may not contain text
            try:
                delta = chunk.choices[0].delta
                # delta.content may be None
                piece = delta.content or ""
            except Exception:
                # If structure differs, try to stringify the chunk
                piece = str(chunk)
            response_text += piece

        return jsonify({"response": response_text, "model_used": model}), 200

    except Exception as e:
        # Return a helpful error message (and print stack to server logs)
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return jsonify({"error": "Server error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
