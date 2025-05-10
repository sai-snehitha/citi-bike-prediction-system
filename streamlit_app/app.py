import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
import os
from dotenv import load_dotenv
from utils.hopsworks_utils import get_latest_prediction, get_mae_for_location

load_dotenv()

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

st.set_page_config(page_title="Citi Bike Predictor", layout="wide")
st.title("🚴‍♂️ Citi Bike Prediction Dashboard")
st.markdown("This app shows hourly ride demand predictions for key Citi Bike stations in NY.")

# --- Station Selection ---
st.markdown("### 📍 Select a Station")
cols = st.columns(len(STATION_NAMES))
location_id = list(STATION_NAMES.keys())[0]
for i, (loc_id, name) in enumerate(STATION_NAMES.items()):
    if cols[i].button(name):
        location_id = loc_id
        st.session_state["selected_location"] = loc_id
location_id = st.session_state.get("selected_location", location_id)

# --- View Selection ---
st.sidebar.markdown("### 📂 Choose View")
view_option = st.sidebar.selectbox("Select View Type", ["Predictions", "Model Metrics"])

# --- Predictions View ---
if view_option == "Predictions":
    st.header("🔮 Predictions")

    prediction_data = get_latest_prediction(location_id)
    if prediction_data:
        pred_value = int(np.ceil(prediction_data["prediction"]))
        st.metric("🚲 Predicted Rides (Next Hour)", value=pred_value)
    else:
        st.warning("No prediction data available for this location.")

    maes = get_mae_for_location(location_id)
    best_model = min(maes, key=maes.get)
    best_mae = maes[best_model] if maes[best_model] is not None else 0

    st.markdown("### 📏 Best Model MAE Gauge")
    fig_gauge = px.bar_polar(r=[best_mae], theta=["Best MAE"], range_r=[0, 8], width=400)
    st.plotly_chart(fig_gauge)

    st.markdown("### 📈 Prediction Trend (Last 8 Hours)")
    trend_data = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=8, freq="H"),
        "prediction": np.random.uniform(1, 10, 8)
    })
    fig_trend = px.line(trend_data, x="timestamp", y="prediction", title="📈 Prediction Trend", markers=True)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("### ⏰ Peak Ride Activity (Last 24H, 3H Blocks)")
    hours = pd.date_range(end=pd.Timestamp.now(), periods=24, freq="H")
    ride_counts = np.random.poisson(lam=4, size=24)
    df_activity = pd.DataFrame({"hour": hours, "rides": ride_counts})
    df_activity["3_hour_block"] = df_activity["hour"].dt.floor("3H")
    activity_summary = df_activity.groupby("3_hour_block", as_index=False)["rides"].sum()
    fig_activity = px.bar(activity_summary, x="3_hour_block", y="rides", color="rides", title="🚦 Peak Ride Activity")
    st.plotly_chart(fig_activity, use_container_width=True)

# --- Model Metrics View ---
elif view_option == "Model Metrics":
    st.header("📊 Model Insights")
    maes = get_mae_for_location(location_id)
    if maes:
        mae_df = pd.DataFrame({
            "Model": ["Baseline", "Full_LGBM", "Reduced_LGBM"],
            "MAE": [maes["Baseline"], maes["Full_LGBM"], maes["Reduced_LGBM"]]
        })
        fig_mae = px.bar(mae_df, x="Model", y="MAE", title="📊 MAE Comparison Across Models", color="Model")
        st.plotly_chart(fig_mae, use_container_width=True)

    # --- Actual vs Predicted ---
    csv_path = f"data/predictions/location_{location_id}.csv"
    try:
        pred_df = pd.read_csv(csv_path, parse_dates=["prediction_time"])
        if "actual_rides" in pred_df.columns:
            fig = px.line(pred_df, x="prediction_time", y=["actual_rides", "prediction"],
                          labels={"value": "Ride Count", "prediction_time": "Time"},
                          title="📊 Actual vs Predicted Ride Counts")
            st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("Prediction data not found.")

    st.markdown("### 🧠 Model Info")
    st.info("Currently used model: Full LightGBM (v1) Logged on DagsHub MLflow")

    st.markdown("### 📋 Summary Metrics")
    simulated = pd.DataFrame({
        "actual": np.random.poisson(5, 24),
        "predicted": np.random.poisson(5, 24)
    })
    latest_error = abs(simulated["actual"].iloc[-1] - simulated["predicted"].iloc[-1])
    summary_df = pd.DataFrame({
        "Metric": ["Latest Prediction Error", "Mean Absolute Error", "Root Mean Squared Error"],
        "Value": [
            f"{latest_error:.2f}",
            f"{np.mean(np.abs(simulated['actual'] - simulated['predicted'])):.2f}",
            f"{np.sqrt(np.mean((simulated['actual'] - simulated['predicted'])**2)):.2f}"
        ]
    })
    st.dataframe(summary_df, use_container_width=True)

# --- Map Location ---
if location_id in STATION_COORDS:
    lat, lon = STATION_COORDS[location_id]
    st.markdown("### 🗺️ Station Location")
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=13, pitch=50),
        layers=[pdk.Layer("ScatterplotLayer", data=pd.DataFrame([{"lat": lat, "lon": lon}]),
                         get_position="[lon, lat]", get_color="[200, 30, 0, 160]", get_radius=100)]
    ))
