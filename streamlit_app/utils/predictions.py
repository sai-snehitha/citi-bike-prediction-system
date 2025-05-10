import pandas as pd

def load_predictions(location_id: str) -> pd.DataFrame:
    # Simulated path (replace with actual S3/URL/Feature Store download if needed)
    path = f"data/predictions/location_{location_id}.csv"
    return pd.read_csv(path, parse_dates=["prediction_datetime"])
