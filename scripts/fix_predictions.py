import os
import pandas as pd

data_path = "data/predictions/"

for file in os.listdir(data_path):
    if file.endswith(".csv") and file.startswith("location_"):
        location_id = file.replace("location_", "").replace(".csv", "")
        file_path = os.path.join(data_path, file)
        df = pd.read_csv(file_path, parse_dates=["prediction_time"])
        df["location_id"] = location_id
        df.to_csv(file_path, index=False)
        print(f"✅ Updated {file} with location_id = {location_id}")
