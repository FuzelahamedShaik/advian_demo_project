import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
OUT = Path(__file__).resolve().parents[1] / 'data'
OUT.mkdir(exist_ok=True)

n = 12000
freq = '1min'
ts = pd.date_range('2026-01-01', periods=n, freq=freq)
cycle = np.sin(np.arange(n) * 2*np.pi / 1440)
shift = ((np.arange(n) // 480) % 3).astype(int)

production_speed = 100 + 8*cycle - 3*(shift == 2) + np.random.normal(0, 1.8, n)
temperature = 62 + 0.08*production_speed + 3*cycle + np.random.normal(0, 1.0, n)
vibration = 0.55 + 0.003*production_speed + np.random.normal(0, 0.04, n)
pressure = 5.4 + 0.015*production_speed + np.random.normal(0, 0.08, n)
motor_current = 21 + 0.12*production_speed + 0.7*vibration + np.random.normal(0, 0.45, n)
energy_consumption = 42 + 0.35*motor_current + 0.08*temperature + np.random.normal(0, 0.8, n)
defect_rate = np.clip(0.018 + 0.0008*(temperature-70) + 0.008*np.maximum(vibration-0.9, 0) + np.random.normal(0, 0.004, n), 0, None)

anomaly_type = np.array(['normal'] * n, dtype=object)
root_cause = np.array(['normal operation'] * n, dtype=object)

def inject(start, end, kind):
    idx = np.arange(start, min(end, n))
    if kind == 'overheating':
        temperature[idx] += np.linspace(8, 22, len(idx)) + np.random.normal(0, 1.0, len(idx))
        motor_current[idx] += np.linspace(1, 5, len(idx))
        energy_consumption[idx] += np.linspace(3, 9, len(idx))
        defect_rate[idx] += np.linspace(0.01, 0.045, len(idx))
        root = 'cooling system degradation or thermal overload'
    elif kind == 'vibration_instability':
        vibration[idx] += np.linspace(0.35, 1.1, len(idx)) + np.random.normal(0, 0.05, len(idx))
        motor_current[idx] += np.linspace(0.8, 3.5, len(idx))
        defect_rate[idx] += np.linspace(0.005, 0.035, len(idx))
        root = 'bearing wear, imbalance, or loose mechanical component'
    elif kind == 'pressure_drop':
        pressure[idx] -= np.linspace(0.7, 2.1, len(idx)) + np.random.normal(0, 0.04, len(idx))
        production_speed[idx] -= np.linspace(2, 12, len(idx))
        defect_rate[idx] += np.linspace(0.006, 0.025, len(idx))
        root = 'leakage, valve issue, or blocked pneumatic/hydraulic line'
    elif kind == 'energy_inefficiency':
        motor_current[idx] += np.linspace(2, 7, len(idx))
        energy_consumption[idx] += np.linspace(5, 18, len(idx))
        production_speed[idx] -= np.linspace(1, 6, len(idx))
        root = 'motor inefficiency, friction, or suboptimal operating regime'
    else:
        return
    anomaly_type[idx] = kind
    root_cause[idx] = root

for start, kind in [(1800,'overheating'),(3600,'vibration_instability'),(5600,'pressure_drop'),(7800,'energy_inefficiency'),(9700,'overheating'),(10800,'vibration_instability')]:
    inject(start, start+220, kind)

df = pd.DataFrame({
    'timestamp': ts,
    'temperature': temperature,
    'vibration': vibration,
    'pressure': pressure,
    'motor_current': motor_current,
    'production_speed': production_speed,
    'energy_consumption': energy_consumption,
    'defect_rate': defect_rate,
    'anomaly_type': anomaly_type,
    'root_cause': root_cause,
})
df.to_csv(OUT / 'industrial_iot_process_data.csv', index=False)
print(f'Wrote {OUT / "industrial_iot_process_data.csv"} with {len(df)} rows')
