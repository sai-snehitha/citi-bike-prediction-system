import hopsworks
import joblib
import pandas as pd
from datetime import datetime
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

# Step 5: Run predictions
predictions = []
prediction_time = datetime.now()

for loc in top_locations:
    latest = df[df["location_id"] == loc].sort_values("pickup_hour").tail(1)
    if latest.empty:
        print(f"⚠️ No data found for location {loc}. Skipping.")
        continue

    X_latest = latest[[col for col in df.columns if "lag_" in col or col in ["hour", "dayofweek", "is_weekend"]]]
    y_pred = model_lgb.predict(X_latest)[0]
    predictions.append((loc, y_pred, prediction_time))

# Step 6: Save to Hopsworks
df_pred = pd.DataFrame(predictions, columns=["location_id", "prediction", "prediction_time"])

pred_fg = fs.get_or_create_feature_group(
    name="citi_bike_predictions",
    version=1,
    primary_key=["location_id", "prediction_time"],
    description="Predicted rides per location",
    event_time="prediction_time"
)
pred_fg.insert(df_pred)

# Step 7: Save to CSVs in data/predictions
os.makedirs("data/predictions", exist_ok=True)

for loc in top_locations:
    loc_df = df_pred[df_pred["location_id"] == loc].copy()
    file_path = f"data/predictions/location_{loc}.csv"

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path, parse_dates=["prediction_time"])
        loc_df = pd.concat([existing_df, loc_df], ignore_index=True).drop_duplicates()

    loc_df.to_csv(file_path, index=False)

print("✅ Predictions inserted into Hopsworks and saved locally.")
