import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / 'models'
SENSORS = ['temperature','vibration','pressure','motor_current','production_speed','energy_consumption','defect_rate']
ACTIONS = {
    'overheating': 'Inspect cooling circuit, reduce load, verify coolant flow and thermal sensors.',
    'vibration_instability': 'Inspect bearings, alignment, fasteners, and rotating components before further operation.',
    'pressure_drop': 'Check valves, leaks, filters, compressors/pumps, and blocked lines.',
    'energy_inefficiency': 'Inspect motor load, friction, lubrication, drive settings, and production speed setpoints.',
    'normal': 'Continue monitoring. No immediate intervention required.'
}

def add_features(df):
    df = df.copy()
    for col in SENSORS:
        df[f'{col}_roll_mean_15'] = df[col].rolling(15, min_periods=1).mean()
        df[f'{col}_roll_std_15'] = df[col].rolling(15, min_periods=1).std().fillna(0)
        df[f'{col}_delta'] = df[col].diff().fillna(0)
    return df

def load_models():
    return joblib.load(MODEL_DIR / 'anomaly_detector.joblib'), joblib.load(MODEL_DIR / 'anomaly_classifier.joblib')

def explain_row(row, baselines, stds, top_n=4):
    impacts = []
    for s in SENSORS:
        z = (row[s] - baselines.get(s, 0)) / max(stds.get(s, 1), 1e-6)
        impacts.append({'sensor': s, 'value': float(row[s]), 'baseline': float(baselines.get(s, 0)), 'z_score': float(z), 'impact': float(abs(z))})
    return sorted(impacts, key=lambda x: x['impact'], reverse=True)[:top_n]

def score_dataframe(df):
    detector, classifier = load_models()
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    feat_df = add_features(df)
    features = detector['features']
    raw_scores = detector['pipeline'].decision_function(feat_df[features])
    is_anomaly = detector['pipeline'].predict(feat_df[features]) == -1
    anomaly_score = 1 / (1 + np.exp(8 * raw_scores))
    anomaly_type = np.array(['normal'] * len(df), dtype=object)
    if is_anomaly.any():
        anomaly_type[is_anomaly] = classifier['pipeline'].predict(feat_df.loc[is_anomaly, classifier['features']])
    explanations, recommendations = [], []
    for pos, (_, row) in enumerate(feat_df.iterrows()):
        typ = anomaly_type[pos]
        explanations.append(explain_row(row, detector['baselines'], detector['stds']))
        recommendations.append(ACTIONS.get(typ, ACTIONS['normal']))
    result = df.copy()
    result['is_anomaly'] = is_anomaly
    result['anomaly_score'] = anomaly_score
    result['predicted_anomaly_type'] = anomaly_type
    result['recommendation'] = recommendations
    result['explanation'] = explanations
    return result
