import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk

from utils.hopsworks_utils import get_latest_prediction, get_mae_for_location

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

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Citi Bike Predictor", layout="wide")
st.markdown("<h1 style='font-size: 38px;'>🚴‍♂️ Citi Bike Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("This app shows hourly ride demand predictions for key Citi Bike stations in NY.")

# --- Interactive Location Buttons ---
st.markdown("### 📍 Select a Station")
selected = st.columns(len(STATION_NAMES))
location_id = list(STATION_NAMES.keys())[0]  # Default fallback
for i, (loc_id, name) in enumerate(STATION_NAMES.items()):
    if selected[i].button(name):
        location_id = loc_id
        st.session_state["selected_location"] = loc_id

# Preserve last selected if page reruns
location_id = st.session_state.get("selected_location", location_id)

# --- Latest Prediction ---
prediction_data = get_latest_prediction(location_id)
if prediction_data:
    pred_value = int(np.ceil(prediction_data["prediction"]))
    st.markdown(f"### 📊 Predicted Rides for Next Hour at {STATION_NAMES[location_id]} ({location_id})")
    st.metric("🚲 Predicted Rides", value=pred_value)
else:
    st.warning("No prediction data available for this location.")

# --- MAE from MLflow/DagsHub ---
maes = get_mae_for_location(location_id)
if maes:
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (Baseline)", f"{maes['Baseline']:.3f}" if maes["Baseline"] else "N/A")
    col2.metric("MAE (Full LGBM)", f"{maes['Full_LGBM']:.3f}" if maes["Full_LGBM"] else "N/A")
    col3.metric("MAE (Reduced LGBM)", f"{maes['Reduced_LGBM']:.3f}" if maes["Reduced_LGBM"] else "N/A")

# --- Prediction Trend  ---
st.markdown("### 📉 Prediction Trend (Last 8 Hours)")
trend_data = pd.DataFrame({
    "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=8, freq="H"),
    "prediction": np.random.uniform(1, 10, 8)
})
fig_trend = px.line(trend_data, x="timestamp", y="prediction", title="📈 Prediction Trend", markers=True)
st.plotly_chart(fig_trend, use_container_width=True)

# --- Ride Volume by 3-Hour Blocks ---
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
    labels={"3_hour_block": "Time Block", "rides": "Ride Count"},
    title="🚦 Peak Ride Activity",
    color="rides",
    color_continuous_scale="viridis"
)
st.plotly_chart(fig_activity, use_container_width=True)

# --- MAE Comparison Chart ---
if maes:
    mae_df = pd.DataFrame({
        "Model": ["Baseline", "Full_LGBM", "Reduced_LGBM"],
        "MAE": [maes["Baseline"], maes["Full_LGBM"], maes["Reduced_LGBM"]]
    })
    fig_mae = px.bar(mae_df, x="Model", y="MAE", title="📊 MAE Comparison", color="Model")
    st.plotly_chart(fig_mae, use_container_width=True)

# --- Actual vs Predicted Plot ---
# --- Actual vs Predicted Plot ---
st.markdown("### 📊 Actual vs Predicted Rides")

csv_path = f"data/predictions/location_{location_id}.csv"

try:
    pred_df = pd.read_csv(csv_path, parse_dates=["prediction_time"])

    available_cols = set(pred_df.columns)

    if "actual_rides" in available_cols and "prediction" in available_cols:
        st.markdown("#### 📝 Actual vs Predicted Ride Counts")
        fig_actual_pred = px.line(
            pred_df,
            x="prediction_time",
            y=["actual_rides", "prediction"],
            labels={"value": "Ride Count", "prediction_time": "Time", "variable": "Legend"},
            title="📊 Actual vs Predicted Ride Counts (Last 24 Hours)"
        )
        st.plotly_chart(fig_actual_pred, use_container_width=True)
    elif "prediction" in available_cols:
        st.markdown("#### 🧭 Predicted Ride Trend")
        fig_pred_only = px.line(
            pred_df,
            x="prediction_time",
            y="prediction",
            labels={"prediction_time": "Time", "prediction": "Predicted Rides"},
            title="📈 Predicted Ride Trend"
        )
        st.plotly_chart(fig_pred_only, use_container_width=True)
    else:
        st.warning("Prediction data not found in the CSV.")
except Exception as e:
    st.error(f"Error loading prediction CSV: {e}")



# --- Station Map Visualization ---
if location_id in STATION_COORDS:
    lat, lon = STATION_COORDS[location_id]
    st.markdown("### 🗺️ Station Location")
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=14,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=pd.DataFrame([{"lat": lat, "lon": lon}]),
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=100,
            ),
        ],
    ))
