import hopsworks
import pandas as pd
import streamlit as st
import os
import mlflow
from mlflow.tracking import MlflowClient

# --- Connect to Hopsworks securely using Streamlit secrets ---


def connect_hopsworks():
    try:
        # ✅ Correctly reference section name in Streamlit secrets
        api_key = st.secrets["HOPSWORKS_NEW"]["api_key"]
        project_name = st.secrets["HOPSWORKS_NEW"]["project"]

        # ✅ Authenticate with Hopsworks
        hopsworks.login(api_key=api_key)
        project = hopsworks.get_project(project_name)
    except Exception as e:
        st.error("❌ Could not authenticate with Hopsworks. Please check your API key and project name in Streamlit secrets.")
        st.stop()

    fs = project.get_feature_store()
    mr = project.get_model_registry()
    return project, fs, mr








# --- Get latest prediction for a given location ---
def get_latest_prediction(location_id: str):
    project, fs, _ = connect_hopsworks()
    fg = fs.get_feature_group("citi_bike_predictions", version=1)
    df = fg.read(read_options={"use_hive": True})
    df_filtered = df[df["location_id"] == location_id].sort_values("prediction_time", ascending=False)

    if df_filtered.empty:
        return None

    latest = df_filtered.iloc[0]
    return {
        "prediction": latest["prediction"],
        "timestamp": pd.to_datetime(latest["prediction_time"], unit='ms')
    }

# --- Fetch MAE from DagsHub MLflow securely via Streamlit secrets ---
def get_mae_for_location(location_id: str):
    os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["DAGSHUB"]["username"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["DAGSHUB"]["token"]

    mlflow.set_tracking_uri("https://dagshub.com/sai-snehitha/citi-bike-prediction-system.mlflow")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("citi-bike-project")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.location_id = '{location_id}'",
        order_by=["start_time DESC"]
    )

    maes = {"Baseline": None, "Full_LGBM": None, "Reduced_LGBM": None}
    for run in runs:
        model_type = run.data.tags.get("model_type")
        if model_type in maes and maes[model_type] is None:
            maes[model_type] = run.data.metrics.get("mae")
        if all(v is not None for v in maes.values()):
            break

    return maes
