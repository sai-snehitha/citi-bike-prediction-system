# streamlit_app/test_secrets.py

import streamlit as st

st.set_page_config(page_title="Secrets Test")

st.title("🔐 Streamlit Secrets Test")

# Test Hopsworks
try:
    api_key = st.secrets["HOPSWORKS_API_KEY"]
    st.success(f"Hopsworks API Key loaded successfully! (First 5 chars: {api_key[:5]}...)")
except Exception as e:
    st.error(f"❌ Failed to load Hopsworks API Key: {e}")

# Test DagsHub
try:
    username = st.secrets["DAGSHUB"]["username"]
    token = st.secrets["DAGSHUB"]["token"]
    st.success(f"DagsHub credentials loaded! Username: {username}, Token starts with: {token[:5]}...")
except Exception as e:
    st.error(f"❌ Failed to load DagsHub credentials: {e}")
