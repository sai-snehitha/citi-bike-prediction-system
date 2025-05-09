# scripts/feature_pipeline.py

import hopsworks
import pandas as pd
from scripts.data_utils import transform_ts_data_into_features_and_target_loop
  

# Step 1: Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Step 2: Load new raw data (this could be S3, CSVs, or dummy rows for automation)
# Replace this with your real data load step
project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group(name="citi_bike_features", version=2)
new_data = fg.read(read_options={"use_hive": True})  # Load data from Hopsworks


# Step 3: Run feature engineering
feature_df = engineer_features(new_data)

# Step 4: Load to Hopsworks feature group
fg = fs.get_or_create_feature_group(
    name="citi_bike_features",
    version=2,
    primary_key=["pickup_hour", "location_id"],
    description="Time series features for Citi Bike predictions"
)

fg.insert(feature_df, write_options={"wait_for_job": True})
print("✅ Feature group loaded to Hopsworks.")
