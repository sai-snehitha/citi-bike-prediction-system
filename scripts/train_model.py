import hopsworks
import joblib
import tempfile
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Step 1: Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

# Step 2: Load feature data
fg = fs.get_feature_group(name="citi_bike_features", version=2)
df = fg.read()

# Step 3: Prepare training set
X = df[[col for col in df.columns if "lag_" in col or col in ["hour", "dayofweek", "is_weekend"]]]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Step 5: Save model to temp dir
model_dir = tempfile.mkdtemp()
joblib.dump(model, f"{model_dir}/model.pkl")

# Step 6: Register to Hopsworks
model_hops = mr.python.create_model(
    name="citi_bike_best_model",
    metrics={"mae": mae},
    description="LightGBM model for Citi Bike predictions",
    input_example=X_train[:2],
    model_dir=model_dir,
    requirements=["lightgbm", "scikit-learn"]
)

model_hops.save()
print("✅ Model trained and saved to Hopsworks.")
