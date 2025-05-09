import hopsworks
import pandas as pd
from scripts.data_utils import transform_ts_data_into_features_and_target_loop

# Step 1: Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Step 2: Load raw time series data from an existing feature group
fg = fs.get_feature_group(name="citi_bike_features", version=2)
ts_df = fg.read(read_options={"use_hive": True})

# Step 3: Extract all unique location IDs from the dataset
location_ids = ts_df.columns[ts_df.columns.str.startswith("JC")]  # update this if needed
# OR more robustly:
location_ids = ts_df.columns.difference(["pickup_hour"])

# Step 4: Run feature engineering
feature_dfs = transform_ts_data_into_features_and_target_loop(ts_df, location_ids)

# Step 5: Combine all location DataFrames into one
feature_df = pd.concat(feature_dfs.values()).reset_index()

# Step 6: Load engineered features into Hopsworks
fg = fs.get_or_create_feature_group(
    name="citi_bike_features",
    version=2,
    primary_key=["pickup_hour", "location_id"],
    description="Time series features for Citi Bike predictions"
)

fg.insert(feature_df, write_options={"wait_for_job": True})
print("✅ All features loaded to Hopsworks.")
