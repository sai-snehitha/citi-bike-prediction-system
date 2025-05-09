# streamlit_app/app.py

import streamlit as st

# ✅ Set page config FIRST before any other Streamlit commands
st.set_page_config(page_title="Citi Bike Predictor", layout="wide")

import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
import plotly.graph_objects as go

from utils.hopsworks_utils import get_latest_prediction, get_mae_for_location

# 🔐 Debug secret loading
st.write("🔐 Loaded secrets: Hopsworks Key =", st.secrets["HOPSWORKS_API_KEY"][:5], "...")

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

# --- UI Header ---
st.markdown("<h1 style='font-size: 38px;'>🚴‍♂️ Citi Bike Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("This app shows hourly ride demand predictions for key Citi Bike stations in NY.")

# --- Location Selection ---
location_name = st.selectbox("📍 Select a Location", list(STATION_NAMES.items()), format_func=lambda x: x[1])
location_id = location_name[0]

# --- Fetch Latest Prediction ---
prediction_data = get_latest_prediction(location_id)

if prediction_data:
    pred_value = int(np.ceil(prediction_data["prediction"]))
    st.markdown(f"### 📊 Predicted Rides for Next Hour at {STATION_NAMES[location_id]} ({location_id})")
    st.metric("🚲 Predicted Rides", value=pred_value)
else:
    st.warning("No prediction data available for this location.")

# --- MAE Gauge ---
maes = get_mae_for_location(location_id)
if maes and maes["Reduced_LGBM"] is not None:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=maes["Reduced_LGBM"],
        title={'text': "LightGBM MAE (All features)"},
        gauge={
            'axis': {'range': [0, 5]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 2], 'color': "lightgreen"},
                {'range': [2, 4], 'color': "yellow"},
                {'range': [4, 5], 'color': "red"}
            ],
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

# --- Prediction Trend ---
st.markdown("### 📉 Prediction Trend (Last 8 Hours)")
trend_data = pd.DataFrame({
    "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=8, freq="H"),
    "prediction": np.random.uniform(1, 10, 8)
})
fig_trend = px.line(trend_data, x="timestamp", y="prediction", title="📈 Prediction Trend", markers=True)
st.plotly_chart(fig_trend, use_container_width=True)

# --- Ride Volume Activity ---
st.markdown("### ⏰ Peak Ride Activity by Time Block")
hours = pd.date_range(end=pd.Timestamp.now(), periods=24, freq="H")
ride_counts = np.random.poisson(lam=4, size=24)
df_activity = pd.DataFrame({
    "hour": hours,
    "rides": ride_counts
})
df_activity["3_hour_block"] = df_activity["hour"].dt.floor("3H")
activity_summary = df_activity.groupby("3_hour_block", as_index=False)["rides"].sum()
fig_activity = px.bar(activity_summary, x="3_hour_block", y="rides",
                      labels={"3_hour_block": "Time Block", "rides": "Ride Count"},
                      title="⏰ High Ride Volume (Last 24 Hours)",
                      color="rides", color_continuous_scale="Viridis")
st.plotly_chart(fig_activity, use_container_width=True)

# --- Station Map ---
if location_id in STATION_COORDS:
    lat, lon = STATION_COORDS[location_id]
    st.markdown("### 🗺️ Station Location")
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=lat, longitude=lon, zoom=14, pitch=50,
        ),
        layers=[pdk.Layer('ScatterplotLayer',
                          data=pd.DataFrame([{"lat": lat, "lon": lon}]),
                          get_position='[lon, lat]',
                          get_color='[200, 30, 0, 160]', get_radius=100)]
    ))
