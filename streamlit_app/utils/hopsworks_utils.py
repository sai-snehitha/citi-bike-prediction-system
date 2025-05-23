import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import hopsworks
import os


def connect_hopsworks():
    os.environ["HOPSWORKS_API_KEY"] = "Bay9NH1XKrLIxD3P.ADKuYP3KuhEpgyPSpQUzollyck1Z6ndplMqjtOZj69rJaeDzRQROCx3II7TwRmWa"
    os.environ["HOPSWORKS_PROJECT"] = "sneha_taxi_project"

    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    return project, fs, mr


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


def get_mae_for_location(location_id: str):
    mlflow.set_tracking_uri("https://dagshub.com/sai-snehitha/citi-bike-prediction-system.mlflow")
    mlflow.set_experiment("citi-bike-project")

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
