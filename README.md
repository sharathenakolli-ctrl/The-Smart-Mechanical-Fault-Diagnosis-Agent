# Smart Mechanical Fault Diagnosis Agent

An AI-powered multi-agent system that analyzes machine symptoms, sensor data, and historical maintenance logs to identify mechanical faults and provide repair guidance — powered by **IBM Granite** via **watsonx.ai**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Pipeline                      │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Agent 1    │→ │   Agent 2    │→ │   Agent 3    │→ │   Agent 4    │ │
│  │  Symptom     │  │   Fault      │  │  Root Cause  │  │   Repair     │ │
│  │  Detection   │  │  Analysis    │  │  Identifier  │  │  Guidance    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │
│         ↑                  ↑                                              │
│    Sensor Data        RAG Knowledge Base                                  │
│    + Symptoms         (TF-IDF Vector Store)                               │
└─────────────────────────────────────────────────────────────┘
                              ↕
                    IBM Granite (watsonx.ai)
```

## Project Structure

```
├── app.py                          # Flask REST API server
├── orchestrator.py                 # Multi-agent pipeline coordinator
├── granite_client.py               # IBM watsonx.ai / Granite LLM client
├── knowledge_base.py               # RAG knowledge base (8 fault records)
├── agents/
│   ├── symptom_detection_agent.py  # Agent 1: Sensor anomaly + symptom profiling
│   ├── fault_analysis_agent.py     # Agent 2: RAG fault matching + ranking
│   ├── root_cause_agent.py         # Agent 3: Root cause analysis
│   └── repair_guidance_agent.py    # Agent 4: Repair procedures
├── dashboard/
│   └── index.html                  # Visualization dashboard (standalone HTML)
├── test_agents.py                  # Test suite
├── requirements.txt
└── config.env                      # API credentials
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure credentials
Edit `config.env` (already populated):
```
WATSONX_API_KEY=<your-key>
WATSONX_PROJECT_ID=<your-project-id>
WATSONX_URL=https://us-south.ml.cloud.ibm.com
IBM_GRANITE_MODEL=ibm/granite-13b-instruct-v2
```

### 3. Run tests
```bash
python test_agents.py
```

### 4. Start the API server
```bash
python app.py
```

### 5. Open the dashboard
Open `dashboard/index.html` in your browser.

---

## API Reference

### POST `/api/diagnose`
Run the full multi-agent diagnosis pipeline.

**Request:**
```json
{
  "equipment_type": "Centrifugal Pump",
  "reported_symptoms": ["high vibration", "unusual noise near bearing housing"],
  "sensor_data": {
    "vibration_rms": 9.2,
    "temperature": 84.0,
    "noise_db": 93.0
  }
}
```

**Response:**
```json
{
  "status": "success",
  "total_elapsed_s": 12.4,
  "summary": {
    "equipment": "Centrifugal Pump",
    "severity": "Critical",
    "primary_fault": "Bearing Failure",
    "confidence": "High",
    "root_cause": "Lubrication breakdown..."
  },
  "symptom_profile": { ... },
  "fault_analysis": { ... },
  "root_cause": { ... },
  "repair_guidance": { ... }
}
```

### GET `/api/demo/<scenario>`
Run a pre-built demo scenario. Available: `pump_bearing`, `gearbox_wear`, `motor_winding`, `pump_cavitation`

### GET `/api/faults/search?q=<query>&top_k=3`
Search the fault knowledge base using semantic similarity.

### GET `/api/faults/<id>`
Get a specific fault record by ID (F001–F008).

---

## Fault Knowledge Base

| ID | Equipment | Fault Type | Severity |
|----|-----------|------------|----------|
| F001 | Centrifugal Pump | Bearing Failure | High |
| F002 | Centrifugal Pump | Cavitation | Medium |
| F003 | Gearbox | Gear Tooth Wear / Pitting | High |
| F004 | Electric Motor | Winding Insulation Failure | Critical |
| F005 | Reciprocating Compressor | Valve Failure | Medium |
| F006 | Rolling Element Bearing | Outer Race Defect | High |
| F007 | Steam Turbine | Rotor Imbalance | High |
| F008 | Hydraulic System | Hydraulic Pump Wear | Medium |

---

## Multi-Agent Pipeline

### Agent 1 – Symptom Detection
- Classifies each sensor reading as `normal`, `warning`, or `alarm` against ISO thresholds
- Determines overall severity (Normal / Warning / Critical)
- Uses Granite to generate a concise technical symptom profile

### Agent 2 – Fault Analysis
- Builds a TF-IDF query from symptoms and anomalies
- Retrieves top-3 matching faults from the knowledge base (RAG)
- Uses Granite to rank and justify the most probable faults

### Agent 3 – Root Cause Identification
- Performs structured Root Cause Analysis (RCA)
- Extracts contributing factors (operational, environmental, maintenance)
- Estimates confidence level: High / Medium / Low

### Agent 4 – Repair Guidance
- Retrieves step-by-step repair procedures from the knowledge base
- Uses Granite to generate a full repair guidance report with:
  - Safety precautions (LOTO, PPE)
  - Tools and materials required
  - Numbered repair procedure (6–10 steps)
  - Verification and testing steps
  - Preventive measures

---

## Tech Stack
- **IBM Granite 13B Instruct v2** via watsonx.ai
- **Flask** REST API
- **scikit-learn** TF-IDF vector store for RAG
- **Vanilla HTML/CSS/JS** dashboard (no external dependencies)
