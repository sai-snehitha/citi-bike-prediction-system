# Citi Bike Trip Prediction System
# 🚴 Citi Bike Demand Forecasting

This project predicts hourly Citi Bike ride demand for key Jersey City stations and monitors model performance.

## 🔧 Project Structure

- `streamlit_app/app.py` – Ride prediction dashboard (for users)
- `streamlit_app/monitoring.py` – Model monitoring dashboard (for developers)
- `utils/` – Helper scripts for Hopsworks + MLflow
- `notebooks/` – Modeling and feature engineering
- `requirements.txt` – For Streamlit Cloud deployment

## 🌐 Live Apps (via Streamlit Community Cloud)

- **Prediction App**: [Link to deploy](https://share.streamlit.io/)
- **Monitoring App**: [Link to deploy](https://share.streamlit.io/)

## 🛠️ Deployment Instructions

1. Fork this repo or upload to your own GitHub account.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and log in.
3. Click **“New app”** and connect this repo.
4. Choose:
   - `streamlit_app/app.py` → for **prediction app**
   - `streamlit_app/monitoring.py` → for **monitoring app**
5. Make sure `requirements.txt` is present at root.
6. Click **Deploy**.

---

