import pandas as pd

def transform_ts_data_into_features_and_target_loop(ts_df, location_ids):
    feature_dfs = {}

    for location_id in location_ids:
        print(f"📍 Processing location: {location_id}")

        if "location_id" not in ts_df.columns:
            raise ValueError(f"'location_id' not found in ts_df.columns")

        df = ts_df[ts_df["location_id"] == location_id].copy()
        df.rename(columns={"target": "target"}, inplace=True)

        # ✅ Ensure datetime index
        df.index = pd.to_datetime(df.index)

        # Generate lag features
        for lag in range(1, 49):
            df[f"lag_{lag}"] = df["target"].shift(lag)

        # Time-based features
        df["hour"] = df.index.hour
        df["dayofweek"] = df.index.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        # ✅ Required by Hopsworks schema
        df["location_id"] = location_id  # as string
        df["pickup_hour"] = df.index     # timestamp column

        df.dropna(inplace=True)
        feature_dfs[location_id] = df

    return feature_dfs
