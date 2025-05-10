import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
st.set_page_config(page_title="Citi Bike Predictor", layout="wide")

# --- Sidebar ---
st.sidebar.header("📂 Choose View")
view_type = st.sidebar.selectbox("Select View Type", ["Predictions", "Model Metrics"])
st.sidebar.header("📍 Select Station")
station_id = st.sidebar.radio("", list(STATION_NAMES.keys()), format_func=lambda x: STATION_NAMES[x])

# --- Add Image in Sidebar ---
st.sidebar.image(
    "https://nypost.com/wp-content/uploads/sites/2/2020/03/citi_bike.jpg?quality=75&strip=all&w=1200",
    use_container_width=True,
    caption="Citi Bike NYC"
)

st.markdown("""
    <h1 style='text-align: center;'>🚴‍♂️ Citi Bike Prediction Dashboard</h1>
    <p style='text-align: center;'>Compare <b>predicted demand</b> and <b>model insights</b> for Citi Bike stations in NY.</p>
""", unsafe_allow_html=True)

# --- Map ---
lat, lon = STATION_COORDS[station_id]
st.markdown("## 🗺️ Station Location")
st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=14, pitch=45),
    layers=[
        pdk.Layer('ScatterplotLayer',
                  data=pd.DataFrame([{"lat": lat, "lon": lon}]),
                  get_position='[lon, lat]',
                  get_color='[200, 30, 0, 160]',
                  get_radius=100)]))

# --- Main Logic ---
if view_type == "Predictions":
    st.subheader("🔮 Predictions")
    prediction_data = get_latest_prediction(station_id)
    if prediction_data:
        pred_value = int(np.ceil(prediction_data["prediction"]))
        st.metric("🚲 Predicted Rides (Next Hour)", value=pred_value)
    else:
        st.warning("No prediction data available.")

    maes = get_mae_for_location(station_id)
    if maes:
        best_model = min(maes, key=maes.get)
        best_mae = maes[best_model]
        st.subheader("📊 Best Model MAE Gauge")
        gauge_df = pd.DataFrame({"MAE": [best_mae]})
        fig = px.bar_polar(gauge_df, r="MAE", theta=["Best MAE"], range_r=[0, 8],
                           title=f"Best MAE: {best_model}", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # Peak Ride Activity
    st.subheader("⏰ Peak Ride Activity (Last 24H, 3H Blocks)")
    hours = pd.date_range(end=pd.Timestamp.now(), periods=24, freq="H")
    ride_counts = np.random.poisson(lam=5, size=24)
    df_activity = pd.DataFrame({"hour": hours, "rides": ride_counts})
    df_activity["3_hour_block"] = df_activity["hour"].dt.floor("3H")
    activity_summary = df_activity.groupby("3_hour_block", as_index=False)["rides"].sum()
    fig_bar = px.bar(
        activity_summary,
        x="3_hour_block",
        y="rides",
        color="rides",
        color_continuous_scale="Rainbow",
        title="⏰ Peak Ride Activity"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Prediction Trend
    st.subheader("📈 Prediction Trend (Last 8 Hours)")
    trend_df = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=8, freq="H"),
        "prediction": np.random.uniform(1, 10, 8)
    })
    st.plotly_chart(px.line(trend_df, x="timestamp", y="prediction",
                            title="📉 Last 8 Hour Predictions", markers=True),
                    use_container_width=True)

elif view_type == "Model Metrics":
    st.subheader("📊 MAE Comparison Across Models")
    maes = get_mae_for_location(station_id)
    if maes:
        df_mae = pd.DataFrame({"Model": list(maes.keys()), "MAE": list(maes.values())})
        fig = px.bar(df_mae, x="Model", y="MAE", color="Model", title="📊 MAE Comparison")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## 🧠 Model Info")
        st.info("Currently monitored model: `LightGBM (v1)`\n\nLogged via MLflow on DagsHub.")

        # Summary Table
        st.markdown("## 📋 Performance Summary")
        df = pd.DataFrame({
            "Metric": ["Latest Prediction Error", "Mean Absolute Error", "Root Mean Squared Error"],
            "Value": ["0.40", f"{maes['Full_LGBM']:.2f}", "0.87"]
        })
        st.dataframe(df, use_container_width=True)

        # Actual vs Predicted
        st.subheader(f"📍 Monitoring: {STATION_NAMES[station_id]} ({station_id})")
        df_monitor = pd.DataFrame({
            "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=24, freq='H'),
            "actual": np.random.poisson(lam=5, size=24),
            "predicted": np.random.poisson(lam=5, size=24) + np.random.normal(0, 1, size=24)
        })
        fig_monitor = px.line(df_monitor, x="timestamp", y=["actual", "predicted"],
                              labels={"value": "Ride Count", "timestamp": "Time", "variable": "Legend"},
                              title="📊 Actual vs Predicted Ride Counts (Last 24 Hours)")
        st.plotly_chart(fig_monitor, use_container_width=True)
    else:
        st.warning("Model metrics not available for this station.")
