"""
RAG Knowledge Base – Fault Records & Maintenance Logs
Uses an in-memory vector store (TF-IDF + cosine similarity) so there are
no heavyweight external dependencies beyond scikit-learn.
"""
from __future__ import annotations
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------------------------------------------------------
# Static fault knowledge base
# ---------------------------------------------------------------------------
FAULT_RECORDS = [
    {
        "id": "F001",
        "equipment": "Centrifugal Pump",
        "fault_type": "Bearing Failure",
        "symptoms": ["high vibration", "unusual noise", "elevated temperature", "excessive noise near bearing housing"],
        "sensor_indicators": {"vibration_rms": ">5 mm/s", "temperature": ">80°C", "noise_db": ">90 dB"},
        "root_cause": "Lubrication breakdown or contamination leading to metal-to-metal contact in bearing races.",
        "severity": "High",
        "repair": [
            "Shut down pump immediately to prevent catastrophic failure.",
            "Inspect bearing for pitting, spalling, or discolouration.",
            "Replace bearing with OEM-specified type and grade.",
            "Flush and replenish lubricant to correct level with correct grade.",
            "Check shaft alignment with dial indicator – realign if deviation >0.05 mm.",
            "Verify coupling condition and replace if worn.",
            "Run for 30 minutes at partial load; monitor vibration and temperature.",
        ],
        "maintenance_logs": [
            {"date": "2023-03-12", "action": "Bearing replaced – pump 3A", "technician": "J. Smith"},
            {"date": "2023-09-05", "action": "Lubrication interval reduced to 500 h", "technician": "R. Patel"},
        ],
    },
    {
        "id": "F002",
        "equipment": "Centrifugal Pump",
        "fault_type": "Cavitation",
        "symptoms": ["crackling noise", "loss of flow", "fluctuating discharge pressure", "pitting on impeller"],
        "sensor_indicators": {"inlet_pressure": "<0.5 bar", "flow_rate": "<60% nominal", "vibration_rms": "2-4 mm/s"},
        "root_cause": "Fluid vaporisation at impeller inlet due to insufficient suction head (NPSH).",
        "severity": "Medium",
        "repair": [
            "Reduce pump speed or throttle discharge valve to increase suction pressure.",
            "Check and clear any blockages in suction piping.",
            "Lower pump installation height or raise fluid reservoir level.",
            "Inspect impeller for erosion pitting; replace if >10% surface area affected.",
            "Verify NPSH available ≥ NPSH required + 0.5 m safety margin.",
            "Install anti-cavitation trim or inducer if systemic issue persists.",
        ],
        "maintenance_logs": [
            {"date": "2023-06-20", "action": "Impeller replaced due to cavitation erosion", "technician": "A. Kumar"},
        ],
    },
    {
        "id": "F003",
        "equipment": "Gearbox",
        "fault_type": "Gear Tooth Wear / Pitting",
        "symptoms": ["high noise level", "increased vibration at gear mesh frequency", "metal particles in oil", "backlash increase"],
        "sensor_indicators": {"vibration_rms": ">8 mm/s", "oil_particle_count": ">100 ppm Fe", "noise_db": ">95 dB"},
        "root_cause": "Fatigue-induced surface contact stress exceeding material limit; inadequate lubrication or overloading.",
        "severity": "High",
        "repair": [
            "Drain gearbox oil and send sample for spectrometric analysis.",
            "Open gearbox and inspect all gear teeth under magnification.",
            "Replace gear set if pitting covers >25% of tooth face.",
            "Check gear alignment and contact pattern using blue marking compound.",
            "Verify oil viscosity and replace with correct ISO VG grade.",
            "Inspect seals and replace to prevent contamination ingress.",
            "Recalculate load factor; reduce if system is overloaded.",
        ],
        "maintenance_logs": [
            {"date": "2022-11-14", "action": "Gear set replaced in main drive gearbox", "technician": "T. Williams"},
            {"date": "2023-04-30", "action": "Oil analysis – Fe 85 ppm, scheduled inspection", "technician": "L. Chen"},
        ],
    },
    {
        "id": "F004",
        "equipment": "Electric Motor",
        "fault_type": "Winding Insulation Failure",
        "symptoms": ["motor overheating", "burning smell", "tripped breaker", "reduced torque"],
        "sensor_indicators": {"winding_temp": ">130°C", "insulation_resistance": "<1 MΩ", "current_draw": ">FLA*1.2"},
        "root_cause": "Thermal aging, moisture ingress, or voltage spikes degrading stator winding insulation.",
        "severity": "Critical",
        "repair": [
            "De-energise and lock-out / tag-out immediately.",
            "Perform megger test: IR > 1 MΩ required; if failed, rewinding is necessary.",
            "Inspect windings for burn marks, discolouration, or carbon tracking.",
            "Check motor cooling fan and clean air passages.",
            "Verify supply voltage balance – unbalance >1% degrades insulation.",
            "Rewind stator with Class F or H insulation if required.",
            "After repair, perform polarisation index (PI) test; PI > 2 indicates good insulation.",
        ],
        "maintenance_logs": [
            {"date": "2023-01-18", "action": "Motor rewound – 45 kW drive pump motor", "technician": "E. Okonkwo"},
        ],
    },
    {
        "id": "F005",
        "equipment": "Reciprocating Compressor",
        "fault_type": "Valve Failure",
        "symptoms": ["loss of capacity", "high discharge temperature", "abnormal valve noise", "increased power consumption"],
        "sensor_indicators": {"discharge_temp": ">180°C", "capacity": "<75% rated", "power_kw": ">110% rated"},
        "root_cause": "Valve plate or spring fatigue causing incomplete sealing and gas blow-by.",
        "severity": "Medium",
        "repair": [
            "Isolate compressor and bleed pressure before opening valve covers.",
            "Remove and inspect valve assemblies for broken plates or springs.",
            "Replace valve plates, springs, and gaskets as a complete kit.",
            "Verify valve seat for scoring; lap seat if minor, replace if deep scoring.",
            "Check lubrication system oil pressure and flow to valve area.",
            "Perform capacity test after reassembly; should reach ≥95% rated capacity.",
        ],
        "maintenance_logs": [
            {"date": "2023-07-08", "action": "All valve kits replaced – compressor C2", "technician": "M. Singh"},
        ],
    },
    {
        "id": "F006",
        "equipment": "Rolling Element Bearing",
        "fault_type": "Outer Race Defect",
        "symptoms": ["periodic impulse vibration", "high kurtosis value", "BPFO frequency peak in spectrum"],
        "sensor_indicators": {"kurtosis": ">4", "vibration_rms": ">6 mm/s", "BPFO_amplitude": "prominent"},
        "root_cause": "Point defect on outer bearing race due to fatigue spalling or impact damage.",
        "severity": "High",
        "repair": [
            "Schedule planned replacement before bearing progresses to catastrophic failure.",
            "Remove bearing using puller – avoid hammering to prevent secondary damage.",
            "Inspect shaft journal for fretting; polish with emery cloth if minor scoring.",
            "Install new bearing with correct interference fit (check OEM tolerance).",
            "Use induction heater for heated installation; do not exceed 110°C.",
            "Re-grease to 30-50% fill; over-greasing causes overheating.",
            "Collect post-installation vibration baseline for trend comparison.",
        ],
        "maintenance_logs": [
            {"date": "2023-05-22", "action": "Outer race defect detected via vibration analysis – bearing replaced", "technician": "D. Brown"},
        ],
    },
    {
        "id": "F007",
        "equipment": "Steam Turbine",
        "fault_type": "Rotor Imbalance",
        "symptoms": ["1x running speed vibration", "elevated bearing temperatures", "whole-body machine vibration"],
        "sensor_indicators": {"1x_amplitude": ">25 µm", "vibration_rms": ">10 mm/s", "phase_angle": "steady"},
        "root_cause": "Mass redistribution due to blade erosion, deposit build-up, or component loss.",
        "severity": "High",
        "repair": [
            "Take vibration measurements at all bearing pedestals and record phase and amplitude.",
            "Perform influence coefficient balancing in-situ if shutdown is not possible.",
            "During outage: inspect rotor blades for erosion, missing segments, or deposits.",
            "Clean rotor of any scale or deposit build-up.",
            "Perform dynamic balancing on balancing machine to ISO 1940 G1.0 grade.",
            "Re-inspect blade root fixings and replace any showing fatigue cracks.",
            "Run-up test with vibration monitoring at all critical speeds.",
        ],
        "maintenance_logs": [
            {"date": "2022-09-10", "action": "Rotor rebalanced after blade erosion found", "technician": "P. Nakamura"},
        ],
    },
    {
        "id": "F008",
        "equipment": "Hydraulic System",
        "fault_type": "Hydraulic Pump Wear",
        "symptoms": ["slow actuator movement", "high oil temperature", "pump noise", "bypass valve opening"],
        "sensor_indicators": {"system_pressure": "<80% rated", "oil_temp": ">75°C", "noise_db": ">85 dB"},
        "root_cause": "Internal leakage due to worn slipper pads, pistons, or barrel in axial piston pump.",
        "severity": "Medium",
        "repair": [
            "Check system pressure against specifications; pressure drop > 20% indicates internal leakage.",
            "Drain hydraulic reservoir and sample fluid for particle count (ISO 4406).",
            "Remove pump and measure clearances using feeler gauge or micrometer.",
            "Replace slipper pads, pistons, and barrel if clearance exceeds OEM spec.",
            "Flush entire hydraulic circuit with clean fluid before reinstallation.",
            "Replace return line filter and breather filter.",
            "Commission with load testing at 25%, 50%, 75%, 100% rated pressure.",
        ],
        "maintenance_logs": [
            {"date": "2023-02-28", "action": "Axial piston pump rebuilt – hydraulic press unit", "technician": "N. Garcia"},
        ],
    },
]


class FaultKnowledgeBase:
    """
    TF-IDF vector store for fault retrieval.
    Provides semantic similarity search over the fault knowledge base.
    """

    def __init__(self):
        self._records = FAULT_RECORDS
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        # Build document corpus: combine symptoms, fault type, and root cause
        corpus = []
        for r in self._records:
            text = " ".join([
                r["fault_type"],
                r["equipment"],
                " ".join(r["symptoms"]),
                r["root_cause"],
            ])
            corpus.append(text)
        self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Return top-k most relevant fault records for a free-text query."""
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_idx:
            rec = dict(self._records[i])
            rec["relevance_score"] = round(float(scores[i]), 4)
            results.append(rec)
        return results

    def get_all_fault_types(self) -> list[str]:
        return [r["fault_type"] for r in self._records]

    def get_by_id(self, fault_id: str) -> dict | None:
        for r in self._records:
            if r["id"] == fault_id:
                return r
        return None


# Singleton
knowledge_base = FaultKnowledgeBase()
