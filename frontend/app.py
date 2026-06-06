import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE / 'ml'))
from inference import score_dataframe, SENSORS

st.set_page_config(page_title='Industrial IoT RCA Dashboard', layout='wide')
st.title('Industrial IoT Anomaly Detection & Root-Cause Dashboard')
st.caption('Demo aligned with Advian: industrial IoT, process data, time-series anomaly detection, RCA, and deployable ML.')

uploaded = st.sidebar.file_uploader('Upload sensor CSV', type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv(BASE / 'data' / 'industrial_iot_process_data.csv')

window = st.sidebar.slider('Rows to analyse', 200, min(len(df), 5000), min(1200, len(df)), 100)
df = df.tail(window).reset_index(drop=True)
scored = score_dataframe(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric('Rows analysed', len(scored))
c2.metric('Anomalies detected', int(scored['is_anomaly'].sum()))
c3.metric('Anomaly rate', f"{scored['is_anomaly'].mean()*100:.1f}%")
top_type = scored.loc[scored['is_anomaly'], 'predicted_anomaly_type'].mode()
c4.metric('Main anomaly type', top_type.iloc[0] if len(top_type) else 'normal')

sensor = st.selectbox('Select sensor signal', SENSORS, index=0)
fig = px.line(scored, x='timestamp', y=sensor, color='predicted_anomaly_type', title=f'{sensor} over time, colored by predicted operating state')
st.plotly_chart(fig, use_container_width=True)

fig2 = px.line(scored, x='timestamp', y='anomaly_score', title='Anomaly score over time')
st.plotly_chart(fig2, use_container_width=True)

st.subheader('Latest detected anomalies')
anoms = scored[scored['is_anomaly']].tail(10)
if anoms.empty:
    st.success('No anomalies detected in the selected window.')
else:
    for _, row in anoms.iloc[::-1].iterrows():
        with st.expander(f"{row['timestamp']} | {row['predicted_anomaly_type']} | score={row['anomaly_score']:.2f}"):
            st.write('Recommended action:', row['recommendation'])
            st.write('Top evidence sensors:')
            st.dataframe(pd.DataFrame(row['explanation']))

st.subheader('Scored data')
st.dataframe(scored.drop(columns=['explanation']), use_container_width=True)
