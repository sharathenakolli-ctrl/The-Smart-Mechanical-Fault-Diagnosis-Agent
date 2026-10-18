"""
Test script – validates the multi-agent pipeline end-to-end.
Run: python test_agents.py
"""
import sys, json

# ── Test 1: Knowledge Base ────────────────────────────────────────────────
print("=" * 60)
print("TEST 1: Knowledge Base RAG Retrieval")
print("=" * 60)
from knowledge_base import knowledge_base
results = knowledge_base.retrieve("high vibration elevated temperature bearing pump", top_k=3)
assert len(results) > 0, "KBase returned no results"
print(f"  [OK] Retrieved {len(results)} fault records")
for r in results:
    print(f"    [{r['id']}] {r['fault_type']} (score: {r['relevance_score']})")

# ── Test 2: Symptom Detection Agent (offline – no Granite call) ───────────
print("\n" + "=" * 60)
print("TEST 2: Sensor Anomaly Classification")
print("=" * 60)
from agents.symptom_detection_agent import _classify_sensor, THRESHOLDS
test_cases = [
    ("vibration_rms",        1.0,  "normal"),
    ("vibration_rms",        3.5,  "warning"),
    ("vibration_rms",        9.0,  "alarm"),
    ("pressure_bar",         2.0,  "normal"),
    ("pressure_bar",         0.6,  "warning"),
    ("pressure_bar",         0.3,  "alarm"),
    ("insulation_resistance",10.0, "normal"),
    ("insulation_resistance", 3.0, "warning"),
    ("insulation_resistance", 0.8, "alarm"),
]
all_pass = True
for key, val, expected in test_cases:
    result = _classify_sensor(key, val)
    ok = result == expected
    all_pass = all_pass and ok
    sym = "[OK]" if ok else "[FAIL]"
    print(f"  {sym} {key}={val} -> {result} (expected {expected})")
assert all_pass, "Sensor classification tests failed"

# ── Test 3: Full pipeline (mocked Granite) ───────────────────────────────
print("\n" + "=" * 60)
print("TEST 3: Full Pipeline (Granite calls mocked)")
print("=" * 60)

# Monkey-patch granite_generate in every agent module to avoid network calls
import granite_client
import agents.symptom_detection_agent as _sda
import agents.fault_analysis_agent    as _faa
import agents.root_cause_agent        as _rca
import agents.repair_guidance_agent   as _rga

_mock = lambda prompt, **kw: (
    "[MOCK] Granite response: This equipment shows signs of bearing failure "
    "due to elevated vibration and temperature.\n"
    "1. Bearing Failure - high correlation with sensor data.\n"
    "2. Outer Race Defect - secondary possibility.\n"
    "3. Gear Tooth Wear - less probable.\n"
    "Contributing factors:\n- Lubrication issue\n- Overloading\n"
    "Confidence: High\nEstimated downtime: 4-6 hours"
)

_real_gen = granite_client.granite_generate
granite_client.granite_generate = _mock
_sda.granite_generate = _mock
_faa.granite_generate = _mock
_rca.granite_generate = _mock
_rga.granite_generate = _mock

from orchestrator import run_diagnosis

test_inputs = [
    {
        "name": "Pump Bearing Failure",
        "equipment_type": "Centrifugal Pump",
        "reported_symptoms": ["high vibration", "unusual noise near bearing housing", "elevated temperature"],
        "sensor_data": {"vibration_rms": 9.2, "temperature": 84.0, "noise_db": 93.0},
    },
    {
        "name": "Gearbox Wear",
        "equipment_type": "Gearbox",
        "reported_symptoms": ["high noise level", "metal particles in oil", "increased backlash"],
        "sensor_data": {"vibration_rms": 10.5, "oil_temp": 72.0, "noise_db": 98.0},
    },
    {
        "name": "Motor Winding Failure",
        "equipment_type": "Electric Motor",
        "reported_symptoms": ["motor overheating", "burning smell", "tripped breaker"],
        "sensor_data": {"temperature": 138.0, "insulation_resistance": 0.5, "current_draw_pct": 128.0},
    },
]

for tc in test_inputs:
    name     = tc.pop("name")
    result   = run_diagnosis(**tc)
    status   = result.get("status")
    ok       = status == "success"
    sym      = "[OK]" if ok else "[FAIL]"
    sev      = result.get("symptom_profile", {}).get("severity_level", "?")
    fault    = result.get("root_cause", {}).get("primary_fault", "?")
    elapsed  = result.get("total_elapsed_s", "?")
    print(f"  {sym} {name}: status={status}, severity={sev}, fault={fault}, time={elapsed}s")
    if not ok:
        print(f"     Errors: {result.get('errors')}")
        sys.exit(1)

# Restore real granite_generate
granite_client.granite_generate = _real_gen

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL TESTS PASSED [OK]")
print("=" * 60)
print("\nTo start the full system:")
print("  1. pip install -r requirements.txt")
print("  2. python app.py")
print("  3. Open dashboard/index.html in your browser")
print("\nAPI Endpoints:")
print("  POST http://localhost:5000/api/diagnose")
print("  GET  http://localhost:5000/api/demo/pump_bearing")
print("  GET  http://localhost:5000/api/faults/search?q=vibration")
