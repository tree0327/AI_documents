import streamlit as st
import sys
import os

st.title("Debug Mode")
st.write(f"Python: {sys.version}")

try:
    import torch
    st.success(f"Torch Imported: {torch.__version__}")
except Exception as e:
    st.error(f"Torch Fail: {e}")

try:
    import joblib
    st.success(f"Joblib Imported: {joblib.__version__}")
except Exception as e:
    st.error(f"Joblib Fail: {e}")
    
try:
    import sklearn
    from sklearn.cluster import KMeans
    st.success(f"Sklearn Imported: {sklearn.__version__}")
except Exception as e:
    st.error(f"Sklearn Fail: {e}")

st.write("--- Loading Models ---")

try:
    scaler = joblib.load('scaler.pkl')
    st.success("Scaler Loaded")
except Exception as e:
    st.error(f"Scaler Fail: {e}")

try:
    rf = joblib.load('rf_model.pkl')
    st.success("RF Model Loaded")
except Exception as e:
    st.error(f"RF Model Fail: {e}")

try:
    import xgboost as xgb_lib
    xgb = xgb_lib.Booster()
    xgb.load_model('xgb_model.json')
    st.success("XGB Model Loaded (JSON)")
except Exception as e:
    st.error(f"XGB Model Fail: {e}")
