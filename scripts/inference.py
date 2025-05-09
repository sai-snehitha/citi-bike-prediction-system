import hopsworks
import pandas as pd
import joblib
import tempfile

# Step 1: Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

# Step 2: Load latest features from Hopsworks
fg = fs.get_feature_group(name="citi_bike_features", version=2)
df = fg.read(read_options={"use_hive": True})

# Target location for prediction (can loop for all locations if needed)
target_location = "JC115"
df_latest = df[df["location_id"] == target_location].sort_values("pickup_hour").tail(1)

# Prepare input
X_latest = df_latest[[col for col in df.columns if "lag_" in col or col in ["hour", "dayofweek", "is_weekend"]]]

# Step 3: Load model from Hopsworks Model Registry
model = mr.get_model(name="citi_bike_best_model", version=1)
model_dir = model.download()
model_path = f"{model_dir}/model.pkl"
loaded_model = joblib.load(model_path)

# Step 4: Run prediction
prediction = loaded_model.predict(X_latest)[0]
print("✅ Prediction:", prediction)

# Step 5: Save prediction back to Hopsworks
pred_fg = fs.get_or_create_feature_group(
    name="citi_bike_predictions",
    version=1,
    primary_key=["location_id", "prediction_time"],
    description="Predicted ride counts for locations",
)

df_pred = pd.DataFrame({
    "location_id": [target_location],
    "prediction_time": [df_latest["pickup_hour"].values[0]],  # or use pd.Timestamp.now()
    "prediction": [prediction]
})

pred_fg.insert(df_pred)
print("📤 Prediction logged to Hopsworks.")
