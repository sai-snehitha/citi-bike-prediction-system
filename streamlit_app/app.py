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

st.set_page_config(page_title="Citi Bike Prediction Dashboard", layout="wide")
st.title("\U0001F6B4️ Citi Bike Prediction Dashboard")
st.markdown("This app displays hourly predictions and model insights for NYC Citi Bike stations.")

# --- Location selection ---
st.markdown("### \U0001F4CD Select a Station")
cols = st.columns(len(STATION_NAMES))
selected_location = st.session_state.get("selected_location", list(STATION_NAMES.keys())[0])
for i, loc in enumerate(STATION_NAMES):
    if cols[i].button(STATION_NAMES[loc]):
        selected_location = loc
        st.session_state["selected_location"] = loc

location_id = selected_location

# Dropdown to toggle between views
st.markdown("### \U0001F4C2 Choose View")
view = st.selectbox("Select View Type", ["Predictions", "Model Metrics"], index=0)

# Load prediction data (if available)
pred_csv_path = f"data/predictions/location_{location_id}.csv"
pred_df = None
try:
    pred_df = pd.read_csv(pred_csv_path, parse_dates=["prediction_time"])
except:
    pass

# Section 1: Predictions View
if view == "Predictions":
    st.markdown("### \U0001F52E Predictions")
    prediction_data = get_latest_prediction(location_id)
    if prediction_data:
        pred_value = int(np.ceil(prediction_data["prediction"]))
        st.metric("Predicted Rides (Next Hour)", pred_value)
    else:
        st.warning("No prediction data available.")

    maes = get_mae_for_location(location_id)
    if maes and maes["Full_LGBM"]:
        fig_gauge = px.line_polar(r=[maes["Full_LGBM"]], theta=["MAE"], line_close=True)
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("### ⏰ Peak Ride Activity (Last 24H, 3H Blocks)")
    hours = pd.date_range(end=pd.Timestamp.now(), periods=24, freq="H")
    ride_counts = np.random.poisson(lam=4, size=24)
    df_activity = pd.DataFrame({"hour": hours, "rides": ride_counts})
    df_activity["3_hour_block"] = df_activity["hour"].dt.floor("3H")
    activity_summary = df_activity.groupby("3_hour_block", as_index=False)["rides"].sum()
    st.plotly_chart(px.bar(activity_summary, x="3_hour_block", y="rides", color="rides"), use_container_width=True)

    st.markdown("### \U0001F4C8 Prediction Trend (Last 8 Hours)")
    trend_data = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=8, freq="H"),
        "prediction": np.random.uniform(1, 10, 8)
    })
    st.plotly_chart(px.line(trend_data, x="timestamp", y="prediction"), use_container_width=True)

# Section 2: Model Metrics View
elif view == "Model Metrics":
    st.markdown("### \U0001F4CA Model Insights")
    maes = get_mae_for_location(location_id)
    if maes:
        fig_mae = px.bar(pd.DataFrame({
            "Model": list(maes.keys()),
            "MAE": list(maes.values())
        }), x="Model", y="MAE", color="Model")
        st.plotly_chart(fig_mae, use_container_width=True)

    if pred_df is not None:
        if "actual_rides" in pred_df.columns:
            st.markdown("### \U0001F4C9 Actual vs Predicted")
            fig = px.line(pred_df, x="prediction_time", y=["actual_rides", "prediction"],
                         labels={"value": "Ride Count", "prediction_time": "Time"})
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### \U0001F4D1 Model Info")
        st.info("Currently monitored model: `LightGBM (v1)`\n\nLogged via MLflow on DagsHub.")

        st.markdown("### \U0001F4CB Performance Summary")
        pred_df = pred_df.tail(24)
        if "actual_rides" in pred_df.columns:
            error = abs(pred_df.actual_rides - pred_df.prediction)
            rmse = np.sqrt(((pred_df.actual_rides - pred_df.prediction)**2).mean())
            summary_df = pd.DataFrame({
                "Metric": ["Latest Prediction Error", "Mean Absolute Error", "Root Mean Squared Error"],
                "Value": [
                    f"{error.iloc[-1]:.2f}",
                    f"{error.mean():.2f}",
                    f"{rmse:.2f}"
                ]
            })
            st.dataframe(summary_df, use_container_width=True)

# Map in background
if location_id in STATION_COORDS:
    lat, lon = STATION_COORDS[location_id]
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=12,
            pitch=45,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=pd.DataFrame([{"lat": lat, "lon": lon}]),
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=100,
            )
        ],
    ))
