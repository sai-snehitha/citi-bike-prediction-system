# scripts/train_model.py

import hopsworks
import mlflow
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Step 1: Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Step 2: Load feature data
fg = fs.get_feature_group(name="citi_bike_features", version=2)
df = fg.read()

# Step 3: Prepare training set
X = df[[col for col in df.columns if "lag_" in col or col in ["hour", "dayofweek", "is_weekend"]]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train and log model
mlflow.set_tracking_uri("https://dagshub.com/sai-snehitha/citi-bike-prediction-system.mlflow")
mlflow.set_registry_uri("https://dagshub.com/sai-snehitha/citi-bike-prediction-system.mlflow")

with mlflow.start_run():
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mlflow.log_metric("mae", mae)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="citi_bike_best_model"
    )

print("✅ Model trained and registered with MLflow.")
