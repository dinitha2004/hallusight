"""
api/app.py
============
HalluSight — Flask REST API

ENDPOINTS:
  POST /detect   — run hallucination detection on a prompt
  GET  /health   — check if server + pipeline are ready
  GET  /status   — pipeline info (model loaded, probe ready)

Run with:
  python api/app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS

from modules.pipeline import initialise_pipeline, run_pipeline, is_ready

# ── Create Flask app ──────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow requests from the frontend HTML file

# ── Initialise pipeline at startup ───────────────────────────
print("Starting HalluSight API server...")
initialise_pipeline(probe_path="sep_probe.pkl")


# ─────────────────────────────────────────────────────────────
# POST /detect
# Body: { "prompt": "Who walked on the moon?", "max_tokens": 100 }
# ─────────────────────────────────────────────────────────────
@app.route("/detect", methods=["POST"])
def detect():
    """Main detection endpoint."""
    data = request.get_json(silent=True) or {}

    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    max_tokens = int(data.get("max_tokens", 100))
    max_tokens = max(10, min(max_tokens, 300))   # clamp 10–300

    try:
        result = run_pipeline(prompt, max_new_tokens=max_tokens)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Quick liveness check."""
    return jsonify({
        "status": "ok" if is_ready() else "loading",
        "pipeline_ready": is_ready(),
    }), 200


# ─────────────────────────────────────────────────────────────
# GET /status
# ─────────────────────────────────────────────────────────────
@app.route("/status", methods=["GET"])
def status():
    """Detailed status for the frontend."""
    return jsonify({
        "pipeline_ready": is_ready(),
        "model": "facebook/opt-1.3b",
        "probe": "sep_probe.pkl",
        "version": "1.0.0",
    }), 200


# ─────────────────────────────────────────────────────────────
# Run server
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════╗")
    print("║   HalluSight API  →  http://localhost:5000 ║")
    print("╚══════════════════════════════════════════╝\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
