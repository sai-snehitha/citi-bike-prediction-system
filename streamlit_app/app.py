import streamlit as st
import pandas as pd
import plotly.express as px
from utils.predictions import load_predictions

st.set_page_config(page_title="Citi Bike Predictor", layout="wide")

st.title("🚴 Citi Bike Prediction Dashboard")
st.markdown("This app shows hourly ride demand predictions for key Citi Bike stations in NY.")

# Dropdown for location
LOCATION_NAMES = {
    "HB102": "Hoboken Terminal - River St & Hudson Pl",
    "HB105": "City Hall - Washington St & 1 St",
    "JC115": "Grove St PATH"
}
location = st.selectbox("📍 Select a Location", options=list(LOCATION_NAMES.values()))

# Map display to ID
location_id = [k for k, v in LOCATION_NAMES.items() if v == location][0]

# Load and show prediction
df = load_predictions(location_id)
fig = px.line(df, x="prediction_datetime", y="predicted_rides", title=f"Predicted Rides - {location}")
st.plotly_chart(fig)
