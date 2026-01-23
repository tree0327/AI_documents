import joblib
from xgboost import XGBClassifier

print("Loading XGB Pickle...")
try:
    xgb_wrapper = joblib.load('xgb_model.pkl')
    print("Loaded Pickle.")
    
    # XGBClassifier in recent versions wraps the booster
    # usage: model.save_model("filename.json")
    
    print("Saving to JSON...")
    # Access the underlying booster to save
    xgb_wrapper.get_booster().save_model('xgb_model.json')
    print("✅ Successfully converted to 'xgb_model.json'")
    
except Exception as e:
    print(f"❌ Error: {e}")
