import hopsworks
import pandas as pd
from scripts.data_utils import transform_ts_data_into_features_and_target_loop

# Step 1: Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Step 2: Load raw time series data from feature store
fg = fs.get_feature_group(name="citi_bike_features", version=2)
ts_df = fg.read(read_options={"use_hive": True})

# Step 3: Run feature engineering for valid location IDs
location_ids = ts_df["location_id"].unique().tolist()  # Dynamically get location IDs
feature_dfs = transform_ts_data_into_features_and_target_loop(ts_df, location_ids=location_ids)
feature_df = pd.concat(feature_dfs.values())

# Step 4: Drop any non-numeric columns (e.g., strings like 'HB102') to avoid Arrow type mismatch
feature_df = feature_df.select_dtypes(include=["number"])

# Step 5: Insert into feature group
fg = fs.get_or_create_feature_group(
    name="citi_bike_features",
    version=2,
    primary_key=["pickup_hour", "location_id"],
    description="Time series features for Citi Bike predictions"
)

fg.insert(feature_df, write_options={"wait_for_job": True})
print("✅ Feature group loaded to Hopsworks.")
