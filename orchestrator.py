"""
Multi-Agent Orchestrator
Coordinates the four agents in a sequential pipeline:
  Symptom Detection → Fault Analysis → Root Cause → Repair Guidance
"""
from __future__ import annotations
import time
import traceback

from agents.symptom_detection_agent import detect_symptoms
from agents.fault_analysis_agent     import analyse_faults
from agents.root_cause_agent         import identify_root_cause
from agents.repair_guidance_agent    import generate_repair_guidance


def run_diagnosis(
    equipment_type: str,
    reported_symptoms: list[str],
    sensor_data: dict[str, float],
) -> dict:
    """
    Full diagnosis pipeline.

    Parameters
    ----------
    equipment_type     : e.g. "Centrifugal Pump"
    reported_symptoms  : list of free-text symptoms
    sensor_data        : sensor name → float value

    Returns
    -------
    dict with full pipeline results and timing metadata
    """
    pipeline_start = time.time()
    pipeline_log   = []
    errors         = []

    def _step(name: str, fn, *args):
        t0 = time.time()
        try:
            result = fn(*args)
            elapsed = round(time.time() - t0, 2)
            pipeline_log.append({"agent": name, "status": "success", "elapsed_s": elapsed})
            return result
        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            tb = traceback.format_exc()
            pipeline_log.append({"agent": name, "status": "error", "elapsed_s": elapsed, "error": str(exc)})
            errors.append({"agent": name, "error": str(exc), "traceback": tb})
            raise

    try:
        # ── Agent 1: Symptom Detection ────────────────────────────────────
        symptom_profile = _step(
            "SymptomDetectionAgent",
            detect_symptoms,
            equipment_type,
            reported_symptoms,
            sensor_data,
        )

        # ── Agent 2: Fault Analysis ───────────────────────────────────────
        fault_analysis = _step(
            "FaultAnalysisAgent",
            analyse_faults,
            symptom_profile,
        )

        # ── Agent 3: Root Cause Identification ───────────────────────────
        root_cause_result = _step(
            "RootCauseAgent",
            identify_root_cause,
            symptom_profile,
            fault_analysis,
        )

        # ── Agent 4: Repair Guidance ──────────────────────────────────────
        repair_guidance = _step(
            "RepairGuidanceAgent",
            generate_repair_guidance,
            symptom_profile,
            fault_analysis,
            root_cause_result,
        )

        total_elapsed = round(time.time() - pipeline_start, 2)

        return {
            "status": "success",
            "total_elapsed_s": total_elapsed,
            "pipeline_log": pipeline_log,
            # ── Individual agent outputs ──────────────────────────────────
            "symptom_profile": symptom_profile,
            "fault_analysis": fault_analysis,
            "root_cause": root_cause_result,
            "repair_guidance": repair_guidance,
            # ── Top-level summary ─────────────────────────────────────────
            "summary": {
                "equipment":      equipment_type,
                "severity":       symptom_profile["severity_level"],
                "primary_fault":  root_cause_result["primary_fault"],
                "confidence":     root_cause_result["confidence"],
                "root_cause":     root_cause_result["root_cause"],
                "top_repair_steps": repair_guidance["repair_steps"][:3],
            },
        }

    except Exception:
        total_elapsed = round(time.time() - pipeline_start, 2)
        return {
            "status": "error",
            "total_elapsed_s": total_elapsed,
            "pipeline_log": pipeline_log,
            "errors": errors,
        }
