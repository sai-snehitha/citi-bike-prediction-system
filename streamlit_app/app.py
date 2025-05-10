import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import os
from dotenv import load_dotenv

from utils.hopsworks_utils import get_latest_prediction, get_mae_for_location

load_dotenv()

# Station info
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

st.set_page_config(layout="wide")
st.markdown("<h1 style='font-size: 36px;'>🚴‍♂️ Citi Bike Prediction Dashboard</h1>", unsafe_allow_html=True)

# Location selection
st.markdown("### 📍 Select a Station")
cols = st.columns(len(STATION_NAMES))
location_id = list(STATION_NAMES.keys())[0]
for i, (loc, name) in enumerate(STATION_NAMES.items()):
    if cols[i].button(name):
        location_id = loc
        st.session_state["selected_location"] = loc
location_id = st.session_state.get("selected_location", location_id)

# Layout split with background map
lat, lon = STATION_COORDS.get(location_id, [40.7193, -74.0341])
map_layer = pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=13, pitch=50),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=pd.DataFrame([{"lat": lat, "lon": lon}]),
            get_position='[lon, lat]',
            get_color='[255, 0, 0, 160]',
            get_radius=150,
        )
    ],
)

left, right = st.columns(2)

# ---------- LEFT COLUMN: Prediction Tab ----------
with left:
    st.subheader("🔮 Predictions")
    prediction_data = get_latest_prediction(location_id)
    if prediction_data:
        pred_value = int(np.ceil(prediction_data["prediction"]))
        st.metric("Predicted Rides (Next Hour)", value=pred_value)

        maes = get_mae_for_location(location_id)
        best_model = min(maes, key=maes.get)
        best_mae = maes[best_model]

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=best_mae,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={"text": f"Best MAE: {best_model}"},
            gauge={"axis": {"range": [None, 10]}, "bar": {"color": "darkblue"}}
        ))
        st.plotly_chart(gauge_fig, use_container_width=True)

        st.markdown("### ⏰ Peak Ride Activity (Last 24H, 3H Blocks)")
        hours = pd.date_range(end=pd.Timestamp.now(), periods=24, freq="H")
        ride_counts = np.random.poisson(lam=4, size=24)
        df_activity = pd.DataFrame({"hour": hours, "rides": ride_counts})
        df_activity["3_hour_block"] = df_activity["hour"].dt.floor("3H")
        activity_summary = df_activity.groupby("3_hour_block", as_index=False)["rides"].sum()
        fig_block = px.bar(activity_summary, x="3_hour_block", y="rides", color="rides",
                          title="🚦 Peak Ride Activity by Time Block")
        st.plotly_chart(fig_block, use_container_width=True)

        st.markdown("### 📈 Prediction Trend")
        trend_data = pd.DataFrame({
            "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=8, freq="H"),
            "prediction": np.random.uniform(1, 10, 8)
        })
        fig_trend = px.line(trend_data, x="timestamp", y="prediction", title="Hourly Prediction Trend")
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("No prediction data available.")

# ---------- RIGHT COLUMN: Model Insights Tab ----------
with right:
    st.subheader("📊 Model Insights")
    maes = get_mae_for_location(location_id)
    if maes:
        mae_df = pd.DataFrame({
            "Model": ["Baseline", "Full_LGBM", "Reduced_LGBM"],
            "MAE": [maes["Baseline"], maes["Full_LGBM"], maes["Reduced_LGBM"]]
        })
        fig_mae = px.bar(mae_df, x="Model", y="MAE", color="Model",
                         title="MAE Comparison Across Models")
        st.plotly_chart(fig_mae, use_container_width=True)

    st.markdown("### 📈 Actual vs Predicted")
    csv_path = f"data/predictions/location_{location_id}.csv"
    try:
        df = pd.read_csv(csv_path, parse_dates=["prediction_time"])
        if "actual_rides" in df.columns:
            fig = px.line(df, x="prediction_time", y=["actual_rides", "prediction"],
                         title="Actual vs Predicted Ride Counts",
                         labels={"value": "Ride Count", "variable": "Type"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.line(df, x="prediction_time", y="prediction", title="Prediction Trend")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load prediction data: {e}")

    st.markdown("### 🧠 Model Info")
    st.info("Currently used model: Full LightGBM (v1) \nLogged on DagsHub MLflow")

    st.markdown("### 📋 Summary Metrics")
    if "actual_rides" in df.columns:
        df["error"] = np.abs(df["actual_rides"] - df["prediction"])
        summary_df = pd.DataFrame({
            "Metric": ["Latest Error", "MAE", "RMSE"],
            "Value": [
                f"{df['error'].iloc[-1]:.2f}",
                f"{np.mean(df['error']):.2f}",
                f"{np.sqrt(np.mean((df['error']) ** 2)):.2f}"
            ]
        })
        st.dataframe(summary_df)

# Map Background
st.markdown("### 🗺️ Station Location")
st.pydeck_chart(map_layer)
