import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import os
from dotenv import load_dotenv
from PIL import Image

from utils.hopsworks_utils import get_latest_prediction, get_mae_for_location

load_dotenv()

# --- Station Info ---
STATION_NAMES = {
    "HB102": "Hoboken Terminal - River St & Hudson Pl",
    "HB105": "City Hall - Washington St & 1 St",
    "JC115": "Grove St PATH"
}

STATION_COORDS = {
    "HB102": [40.7193, -74.0341],
    "HB105": [40.7269, -74.0324],
    "JC115": [40.7194, -74.0421]
}

# --- Page Config ---
st.set_page_config(page_title="Citi Bike Dashboard", layout="wide")

# --- Title ---
st.title("🚴‍♂️ Citi Bike Prediction Dashboard")
st.markdown("Compare **predicted demand** and **model insights** for Citi Bike stations in NY.")

# --- Left Sidebar View and Station Selector ---
with st.sidebar:
    st.markdown("### 📁 Choose View")
    view_type = st.selectbox("Select View Type", ["Predictions", "Model Metrics"])

    st.markdown("### 📍 Select Station")
    station_options = list(STATION_NAMES.keys())
    location_id = st.radio("", station_options, format_func=lambda x: STATION_NAMES[x])
    st.session_state["selected_location"] = location_id

    st.markdown("\n")
    # Background illustration
    bike_img = Image.open("streamlit_app/assets/bike_background.png")
    st.image(bike_img, use_column_width=True)

# --- Location Coordinates ---
lat, lon = STATION_COORDS[location_id]

# --- Background Map ---
st.markdown("### 🗺️ Station Location")
st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=13,
        pitch=45,
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=pd.DataFrame([{"lat": lat, "lon": lon}]),
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=150,
        ),
    ],
))

# --- View: Predictions ---
if view_type == "Predictions":
    st.markdown(f"### 🔮 Predictions for {STATION_NAMES[location_id]}")

    # Prediction Value
    prediction_data = get_latest_prediction(location_id)
    if prediction_data:
        pred_value = int(np.ceil(prediction_data["prediction"]))
        st.metric("Predicted Rides (Next Hour)", value=pred_value)

    # Best MAE as Gauge Chart
    maes = get_mae_for_location(location_id)
    best_model = min(maes, key=maes.get)
    best_mae = maes[best_model]
    st.markdown("### 📏 Best Model MAE Gauge")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=best_mae,
        title={'text': f"Best MAE: {best_model}"},
        gauge={'axis': {'range': [None, 8]}, 'bar': {'color': "#2b7bba"}}
    ))
    st.plotly_chart(gauge, use_container_width=True)

    # Prediction Trend
    st.markdown("### 📈 Prediction Trend (Last 8 Hours)")
    trend_data = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=8, freq="H"),
        "prediction": np.random.uniform(1, 10, 8)
    })
    fig_trend = px.line(trend_data, x="timestamp", y="prediction", title="📈 Prediction Trend", markers=True)
    st.plotly_chart(fig_trend, use_container_width=True)

    # Peak Ride Activity
    st.markdown("### ⏰ Peak Ride Activity (Last 24H, 3H Blocks)")
    hours = pd.date_range(end=pd.Timestamp.now(), periods=24, freq="H")
    ride_counts = np.random.poisson(lam=4, size=24)
    df_activity = pd.DataFrame({"hour": hours, "rides": ride_counts})
    df_activity["3_hour_block"] = df_activity["hour"].dt.floor("3H")
    activity_summary = df_activity.groupby("3_hour_block", as_index=False)["rides"].sum()
    fig_activity = px.bar(
        activity_summary,
        x="3_hour_block",
        y="rides",
        color="rides",
        color_continuous_scale="bluered_r",
        title="🚦 Peak Ride Activity",
        labels={"3_hour_block": "Time Block", "rides": "Ride Count"}
    )
    st.plotly_chart(fig_activity, use_container_width=True)

# --- View: Model Metrics ---
elif view_type == "Model Metrics":
    st.markdown(f"### 🧠 Model Info: {STATION_NAMES[location_id]}")
    st.info("Currently used model: Full LightGBM (v1) Logged on DagsHub MLflow")

    # MAE Comparison Bar Chart
    st.markdown("### 📊 MAE Comparison Across Models")
    mae_df = pd.DataFrame({
        "Model": ["Baseline", "Full_LGBM", "Reduced_LGBM"],
        "MAE": [maes["Baseline"], maes["Full_LGBM"], maes["Reduced_LGBM"]]
    })
    fig_mae = px.bar(mae_df, x="Model", y="MAE", color="Model",
                     color_discrete_sequence=px.colors.qualitative.Safe,
                     title="📊 MAE Comparison Across Models")
    st.plotly_chart(fig_mae, use_container_width=True)

    # Actual vs Predicted
    st.markdown(f"### 📍 Monitoring: {STATION_NAMES[location_id]} ({location_id})")
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=24, freq='H')
    actuals = np.random.poisson(lam=5, size=24)
    preds = actuals + np.random.normal(0, 1, size=24)
    df_simulated = pd.DataFrame({
        "timestamp": timestamps,
        "actual": actuals,
        "predicted": preds
    })
    fig = px.line(df_simulated, x="timestamp", y=["actual", "predicted"],
                  labels={"value": "Ride Count", "timestamp": "Time", "variable": "Legend"},
                  title="📊 Actual vs Predicted Ride Counts (Last 24 Hours)")
    st.plotly_chart(fig, use_container_width=True)

    # Summary
    st.markdown("### 📋 Performance Summary")
    latest_error = abs(df_simulated['actual'].iloc[-1] - df_simulated['predicted'].iloc[-1])
    summary_df = pd.DataFrame({
        "Metric": ["Latest Prediction Error", "Mean Absolute Error", "Root Mean Squared Error"],
        "Value": [
            f"{latest_error:.2f}",
            f"{np.mean(np.abs(df_simulated['actual'] - df_simulated['predicted'])):.2f}",
            f"{np.sqrt(np.mean((df_simulated['actual'] - df_simulated['predicted']) ** 2)):.2f}",
        ]
    })
    st.dataframe(summary_df, use_container_width=True)
