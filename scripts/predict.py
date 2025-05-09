# predict.py
import hopsworks
import mlflow
import pandas as pd

# Step 1: Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Step 2: Load latest features from Hopsworks
fg = fs.get_feature_group(name="citi_bike_features", version=2)
df = fg.read(read_options={"use_hive": True})

# Change target location as needed
target_location = "JC115"
df_latest = df[df["location_id"] == target_location].sort_values("pickup_hour").tail(1)

# Prepare input
X_latest = df_latest[[col for col in df.columns if "lag_" in col or col in ["hour", "dayofweek", "is_weekend"]]]

# Step 3: Load best model from MLflow
mlflow.set_tracking_uri("https://dagshub.com/sai-snehitha/citi-bike-prediction-system.mlflow")
mlflow.set_registry_uri("https://dagshub.com/sai-snehitha/citi-bike-prediction-system.mlflow")

model_name = "citi_bike_best_model"
model_uri = f"models:/{model_name}/latest"

loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)

# Step 4: Make prediction
prediction = loaded_model.predict(X_latest)[0]
print("✅ Prediction:", prediction)

# Step 5: Log prediction to Hopsworks (optional)
pred_fg = fs.get_or_create_feature_group(
    name="citi_bike_predictions",
    version=1,
    primary_key=["location_id"],
    description="Predicted rides per location"
)

df_pred = pd.DataFrame({
    "location_id": [target_location],
    "prediction": [prediction]
})

pred_fg.insert(df_pred)
