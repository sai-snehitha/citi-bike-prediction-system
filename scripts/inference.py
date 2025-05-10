import hopsworks
import joblib
import pandas as pd
from datetime import datetime, timedelta
import os

# Step 1: Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

# Step 2: Load latest feature data
fg = fs.get_feature_group(name="citi_bike_features", version=2)
df = fg.read(read_options={"use_hive": True})

# Step 3: Define top 3 location_ids
top_locations = ["HB102", "HB105", "JC115"]

# Step 4: Load latest model
model = mr.get_model("citi_bike_best_model", version=1)
model_dir = model.download()
model_path = model_dir + "/model.pkl"
model_lgb = joblib.load(model_path)

# Step 5: Run predictions for past 24 hours
cutoff_time = pd.Timestamp.utcnow() - pd.Timedelta(days=7)


predictions = []

for loc in top_locations:
    df_loc = df[(df["location_id"] == loc) & (df["pickup_hour"] >= cutoff_time)].sort_values("pickup_hour")

    if df_loc.empty:
        print(f"⚠️ No recent data for {loc}. Skipping.")
        continue

    feature_cols = [col for col in df.columns if "lag_" in col or col in ["hour", "dayofweek", "is_weekend"]]
    X = df_loc[feature_cols]
    y_preds = model_lgb.predict(X)

    for ts, pred in zip(df_loc["pickup_hour"], y_preds):
        predictions.append((loc, pred, ts))

# Step 6: Format and save predictions
df_pred = pd.DataFrame(predictions, columns=["location_id", "prediction", "prediction_time"])

# Save to local CSVs per station
for loc in top_locations:
    loc_df = df_pred[df_pred["location_id"] == loc].copy()
    file_path = f"data/predictions/location_{loc}.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path, parse_dates=["prediction_time"])
        loc_df = pd.concat([existing_df, loc_df], ignore_index=True).drop_duplicates()

    loc_df.to_csv(file_path, index=False)

print("✅ Multi-hour predictions saved locally.")
