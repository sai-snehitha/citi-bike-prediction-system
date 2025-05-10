import hopsworks
import pandas as pd
import joblib
from datetime import datetime, timedelta

# Step 1: Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

# Step 2: Load latest feature data
fg = fs.get_feature_group(name="citi_bike_features", version=2)
df = fg.read(read_options={"use_hive": True})

# Step 3: Define top locations
top_locations = ["HB102", "HB105", "JC115"]

# Step 4: Load latest model
model = mr.get_model("citi_bike_best_model", version=1)
model_dir = model.download()
model_path = model_dir + "/model.pkl"
model_lgb = joblib.load(model_path)

# Step 5: Predict for each location, even if no recent data
predictions = []
current_time = pd.Timestamp.utcnow()

for loc in top_locations:
    df_loc = df[df["location_id"] == loc].sort_values("pickup_hour", ascending=False).head(1)

    if df_loc.empty:
        print(f"⚠️ No data found for {loc}. Generating dummy prediction.")
        pred_time = current_time.round("H")
        predictions.append((loc, -1, pred_time))  # fallback dummy prediction
        continue

    feature_cols = [col for col in df.columns if "lag_" in col or col in ["hour", "dayofweek", "is_weekend"]]
    X = df_loc[feature_cols]
    y_pred = model_lgb.predict(X)[0]
    pred_time = df_loc["pickup_hour"].values[0]

    predictions.append((loc, y_pred, pred_time))

# Step 6: Insert to Hopsworks feature group
df_pred = pd.DataFrame(predictions, columns=["location_id", "prediction", "prediction_time"])

# Insert into Hopsworks prediction feature group
fg_pred = fs.get_feature_group(name="citi_bike_predictions", version=1)
fg_pred.insert(df_pred, write_options={"wait_for_job": True})

print("✅ Predictions inserted into Hopsworks.")
