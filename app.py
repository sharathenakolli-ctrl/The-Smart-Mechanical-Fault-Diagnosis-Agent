"""
Flask REST API – Smart Mechanical Fault Diagnosis Agent
Exposes endpoints for the diagnosis pipeline and dashboard data.
"""
from __future__ import annotations
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

from orchestrator import run_diagnosis
from knowledge_base import knowledge_base

app = Flask(__name__)
CORS(app)


# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Smart Mechanical Fault Diagnosis Agent"})


# ── Main diagnosis endpoint ───────────────────────────────────────────────────
@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    """
    POST body (JSON):
    {
        "equipment_type": "Centrifugal Pump",
        "reported_symptoms": ["high vibration", "unusual noise"],
        "sensor_data": {
            "vibration_rms": 8.5,
            "temperature": 82.0,
            "noise_db": 92.0
        }
    }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    equipment_type    = data.get("equipment_type", "Unknown")
    reported_symptoms = data.get("reported_symptoms", [])
    sensor_data       = data.get("sensor_data", {})

    # Convert all sensor values to float
    sensor_data = {k: float(v) for k, v in sensor_data.items()}

    result = run_diagnosis(equipment_type, reported_symptoms, sensor_data)
    return jsonify(result)


# ── Knowledge base retrieval endpoint ────────────────────────────────────────
@app.route("/api/faults/search", methods=["GET"])
def search_faults():
    query = request.args.get("q", "")
    top_k = int(request.args.get("top_k", 3))
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    results = knowledge_base.retrieve(query, top_k=top_k)
    return jsonify({"results": results, "count": len(results)})


# ── Get all fault types ───────────────────────────────────────────────────────
@app.route("/api/faults", methods=["GET"])
def list_faults():
    return jsonify({"fault_types": knowledge_base.get_all_fault_types()})


# ── Get fault by ID ───────────────────────────────────────────────────────────
@app.route("/api/faults/<fault_id>", methods=["GET"])
def get_fault(fault_id: str):
    fault = knowledge_base.get_by_id(fault_id.upper())
    if fault is None:
        return jsonify({"error": f"Fault ID '{fault_id}' not found"}), 404
    return jsonify(fault)


# ── Demo scenarios (pre-built test cases) ────────────────────────────────────
DEMO_SCENARIOS = {
    "pump_bearing": {
        "equipment_type": "Centrifugal Pump",
        "reported_symptoms": [
            "high vibration", "unusual noise near bearing housing", "elevated temperature"
        ],
        "sensor_data": {
            "vibration_rms": 9.2,
            "temperature": 84.0,
            "noise_db": 93.0
        },
    },
    "gearbox_wear": {
        "equipment_type": "Gearbox",
        "reported_symptoms": [
            "high noise level", "metal particles in oil", "increased backlash"
        ],
        "sensor_data": {
            "vibration_rms": 10.5,
            "oil_temp": 72.0,
            "noise_db": 98.0
        },
    },
    "motor_winding": {
        "equipment_type": "Electric Motor",
        "reported_symptoms": [
            "motor overheating", "burning smell", "tripped breaker"
        ],
        "sensor_data": {
            "temperature": 138.0,
            "insulation_resistance": 0.5,
            "current_draw_pct": 128.0
        },
    },
    "pump_cavitation": {
        "equipment_type": "Centrifugal Pump",
        "reported_symptoms": [
            "crackling noise", "loss of flow", "fluctuating discharge pressure"
        ],
        "sensor_data": {
            "pressure_bar": 0.3,
            "vibration_rms": 3.2,
            "noise_db": 82.0
        },
    },
}


@app.route("/api/demo/<scenario>", methods=["GET"])
def run_demo(scenario: str):
    sc = DEMO_SCENARIOS.get(scenario)
    if sc is None:
        return jsonify({
            "error": f"Unknown demo scenario '{scenario}'",
            "available": list(DEMO_SCENARIOS.keys()),
        }), 404
    result = run_diagnosis(**sc)
    return jsonify(result)


@app.route("/api/demo", methods=["GET"])
def list_demos():
    return jsonify({"scenarios": list(DEMO_SCENARIOS.keys())})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)
