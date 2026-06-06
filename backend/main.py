from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE / 'ml'))
from inference import score_dataframe, SENSORS

app = FastAPI(title='Industrial IoT Anomaly Detection API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

class SensorRow(BaseModel):
    timestamp: str
    temperature: float
    vibration: float
    pressure: float
    motor_current: float
    production_speed: float
    energy_consumption: float
    defect_rate: float

@app.get('/')
def root():
    return {'service': 'Industrial IoT Anomaly Detection API', 'endpoints': ['/health','/sample','/predict','/predict-csv']}

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.get('/sample')
def sample(limit: int = 200):
    df = pd.read_csv(BASE / 'data' / 'industrial_iot_process_data.csv').tail(limit)
    scored = score_dataframe(df)
    return scored.drop(columns=['explanation']).to_dict(orient='records')

@app.post('/predict')
def predict(rows: list[SensorRow]):
    df = pd.DataFrame([r.model_dump() for r in rows])
    scored = score_dataframe(df)
    return scored.to_dict(orient='records')

@app.post('/predict-csv')
async def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    scored = score_dataframe(df)
    return scored.to_dict(orient='records')
