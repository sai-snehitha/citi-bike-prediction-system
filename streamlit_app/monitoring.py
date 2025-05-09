# streamlit_app/monitoring.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

STATION_NAMES = {
    "HB102": "Hoboken Terminal - River St & Hudson Pl",
    "HB105": "City Hall - Washington St & 1 St",
    "JC115": "Grove St PATH"
}

def load_simulated_data(location_id):
    np.random.seed(hash(location_id) % 123456)
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=24, freq='H')
    actuals = np.random.poisson(lam=5, size=24)
    preds = actuals + np.random.normal(0, 1, size=24)
    return pd.DataFrame({
        "timestamp": timestamps,
        "actual": actuals,
        "predicted": preds
    })

st.set_page_config(page_title="Citi Bike Monitoring", layout="wide")
st.title("📉 Citi Bike Model Monitoring Dashboard")
st.markdown("Compare **actual** vs **predicted** ride counts for key stations and assess model behavior.")

st.markdown("### 🚲 Select a Station to Monitor")
cols = st.columns(len(STATION_NAMES))
selected_location = st.session_state.get("selected_location", None)
for i, loc_id in enumerate(STATION_NAMES):
    if cols[i].button(STATION_NAMES[loc_id]):
        selected_location = loc_id
        st.session_state["selected_location"] = loc_id

if selected_location:
    st.subheader(f"📍 Monitoring: {STATION_NAMES[selected_location]} ({selected_location})")
    df = load_simulated_data(selected_location)

    # Actual vs Predicted
    fig = px.line(df, x="timestamp", y=["actual", "predicted"],
                  labels={"value": "Ride Count", "timestamp": "Time", "variable": "Legend"},
                  title="📊 Actual vs Predicted Ride Counts (Last 24 Hours)")
    st.plotly_chart(fig, use_container_width=True)

    # MAE Trend
    st.markdown("### 📉 MAE Trend Over Time")
    days = pd.date_range(end=pd.Timestamp.today(), periods=7)
    mae_trend = pd.DataFrame({
        "date": days,
        "MAE": np.random.uniform(1.5, 3.0, size=7)
    })
    fig_mae = px.line(mae_trend, x="date", y="MAE", title="📊 MAE Trend (Past Week)")
    st.plotly_chart(fig_mae, use_container_width=True)

    # MAE Comparison
    st.markdown("### 📊 MAE Comparison Across Models")
    mae_df = pd.DataFrame({
        "Model": ["Baseline", "Full_LGBM", "Reduced_LGBM"],
        "MAE": [3.65, 1.83, 1.87]
    })
    fig_comp = px.bar(mae_df, x="Model", y="MAE", title="📊 MAE Comparison Across Models", color="Model")
    st.plotly_chart(fig_comp, use_container_width=True)

    # MAE Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (Baseline)", "3.654")
    col2.metric("MAE (Full LGBM)", "1.834")
    col3.metric("MAE (Reduced LGBM)", "1.867")

    
    # Model Info
    st.markdown("### 🧾 Model Information")
    st.info("Currently monitored model: `LightGBM (v1)`\n\nLogged via MLflow on DagsHub.")

    # Summary
    st.markdown("### 📋 Performance Summary")
    latest_error = abs(df['actual'].iloc[-1] - df['predicted'].iloc[-1])
    summary_df = pd.DataFrame({
        "Metric": ["Latest Prediction Error", "Mean Absolute Error", "Root Mean Squared Error"],
        "Value": [
            f"{latest_error:.2f}",
            f"{np.mean(np.abs(df['actual'] - df['predicted'])):.2f}",
            f"{np.sqrt(np.mean((df['actual'] - df['predicted']) ** 2)):.2f}",
        ]
    })
    st.dataframe(summary_df, use_container_width=True)
else:
    st.info("Please click on a station to begin monitoring.")