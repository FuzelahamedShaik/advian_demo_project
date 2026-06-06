import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data' / 'industrial_iot_process_data.csv'
MODEL_DIR = BASE / 'models'
MODEL_DIR.mkdir(exist_ok=True)

SENSORS = ['temperature','vibration','pressure','motor_current','production_speed','energy_consumption','defect_rate']

def add_features(df):
    df = df.copy()
    for col in SENSORS:
        df[f'{col}_roll_mean_15'] = df[col].rolling(15, min_periods=1).mean()
        df[f'{col}_roll_std_15'] = df[col].rolling(15, min_periods=1).std().fillna(0)
        df[f'{col}_delta'] = df[col].diff().fillna(0)
    return df

df = pd.read_csv(DATA, parse_dates=['timestamp'])
df = add_features(df)
features = [c for c in df.columns if c not in ['timestamp','anomaly_type','root_cause']]

normal = df[df['anomaly_type'] == 'normal']
iso = Pipeline([
    ('scaler', StandardScaler()),
    ('model', IsolationForest(n_estimators=250, contamination=0.09, random_state=42))
])
iso.fit(normal[features])

label_df = df[df['anomaly_type'] != 'normal'].copy()
X_train, X_test, y_train, y_test = train_test_split(
    label_df[features], label_df['anomaly_type'], test_size=0.25, random_state=42, stratify=label_df['anomaly_type']
)
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'))
])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred, labels=list(clf.classes_)).tolist()

baselines = normal[features].mean().to_dict()
stds = normal[features].std().replace(0, 1).to_dict()

joblib.dump({'pipeline': iso, 'features': features, 'baselines': baselines, 'stds': stds}, MODEL_DIR / 'anomaly_detector.joblib')
joblib.dump({'pipeline': clf, 'features': features, 'classes': list(clf.classes_)}, MODEL_DIR / 'anomaly_classifier.joblib')
with open(MODEL_DIR / 'metrics.json', 'w') as f:
    json.dump({'classification_report': report, 'confusion_matrix': cm, 'labels': list(clf.classes_)}, f, indent=2)
print('Models saved to', MODEL_DIR)
print(json.dumps(report, indent=2)[:1000])
