# Edge-Aware Industrial Process Monitoring

A portfolio demo for an **Advian Machine Learning Engineer** role: industrial IoT anomaly detection, root-cause investigation, and production-ready deployment.

## What this demo shows

This project simulates a manufacturing machine with multivariate process signals:

- temperature
- vibration
- pressure
- motor current
- production speed
- energy consumption
- product defect rate

The system ingests time-series process data, detects abnormal machine behavior, classifies the anomaly type, identifies the most likely root cause, shows explainable sensor evidence, and recommends an operational action.

## Why this matches Advian

Advian’s role focuses on:

- industrial IoT and process data
- time-series analysis
- anomaly detection
- root-cause investigations
- deploying ML models into real operations
- practical technical decisions under industrial constraints

This demo directly maps to those needs.

## Architecture

```text
Synthetic Industrial IoT Data
        ↓
Feature Engineering
rolling means, rolling std, deltas
        ↓
Anomaly Detection
Isolation Forest trained on normal operation
        ↓
Anomaly Classification
Random Forest classifier for anomaly type
        ↓
Root-Cause Evidence
sensor deviation / z-score ranking
        ↓
Operational Recommendation
maintenance/action guidance
        ↓
FastAPI + Streamlit Dashboard
```

## Anomaly classes

| Anomaly type | Likely root cause | Operational action |
|---|---|---|
| overheating | Cooling degradation or thermal overload | Inspect cooling circuit and reduce load |
| vibration_instability | Bearing wear, imbalance, loose component | Inspect bearings, alignment, fasteners |
| pressure_drop | Leakage, valve issue, blocked line | Check valves, leaks, filters, pumps |
| energy_inefficiency | Motor inefficiency, friction, bad setpoint | Inspect motor load, lubrication, drive settings |

## Run locally

### 1. Create data

```bash
python ml/generate_data.py
```

### 2. Train models

```bash
python ml/train_model.py
```

### 3. Run the dashboard

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

### 4. Run the API

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Then open:

```text
http://localhost:8000/docs
```

## API examples

Health check:

```bash
curl http://localhost:8000/health
```

Get scored sample rows:

```bash
curl "http://localhost:8000/sample?limit=50"
```

## Docker

Build and run API:

```bash
docker build -t industrial-ml-demo .
docker run -p 8000:8000 industrial-ml-demo
```

## Talking points for interview

1. **Industrial problem framing**  
   The demo is not just anomaly detection. It links anomalies to likely causes and operational action.

2. **Real-world constraints**  
   The model uses interpretable features and lightweight algorithms suitable for edge or near-edge deployment.

3. **False alarm control**  
   Isolation Forest contamination can be tuned based on plant tolerance for missed anomalies vs false alerts.

4. **Explainability**  
   Sensor-level z-score evidence shows which signals changed most strongly compared with normal operation.

5. **Production readiness**  
   The model is exposed through FastAPI and shown in a dashboard, making it easier to integrate with factory systems.

6. **Next improvements**  
   Add drift monitoring, online learning, PLC/SCADA ingestion, MLflow registry, and real maintenance labels.