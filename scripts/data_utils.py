import pandas as pd

def transform_ts_data_into_features_and_target_loop(ts_df, location_ids):
    feature_dfs = {}

    for location_id in location_ids:
        print(f"📊 Processing location: {location_id}")

        df = ts_df[ts_df["location_id"] == location_id].copy()
        df.rename(columns={location_id: "target"}, inplace=True)

        # ✅ Convert pickup_hour to datetime and set as index
        df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
        df.set_index("pickup_hour", inplace=True)

        # 🕓 Create lag features
        for lag in range(1, 49):
            df[f"lag_{lag}"] = df["target"].shift(lag)

        # 🕗 Add time-based features
        df["hour"] = df.index.hour
        df["dayofweek"] = df.index.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        # 🧹 Drop rows with NaNs due to shifting
        df.dropna(inplace=True)

        feature_dfs[location_id] = df

    return feature_dfs

