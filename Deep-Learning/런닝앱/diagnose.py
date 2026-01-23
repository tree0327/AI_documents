import sys
import platform
print(f"Python Version: {sys.version}")
print(f"Platform: {platform.platform()}")

try:
    import numpy
    print(f"Numpy: {numpy.__version__}")
except ImportError:
    print("Numpy not installed")

try:
    import sklearn
    print(f"Sklearn: {sklearn.__version__}")
except ImportError:
    print("Sklearn not installed")

try:
    import torch
    print(f"Torch: {torch.__version__}")
except ImportError:
    print("Torch not installed")

try:
    import joblib
    print(f"Joblib: {joblib.__version__}")
except ImportError:
    print("Joblib not installed")

print("\n--- Testing Model Load ---")
try:
    rf = joblib.load('rf_model.pkl')
    print("Random Forest Loaded successfully")
    xgb = joblib.load('xgb_model.pkl')
    print("XGBoost Loaded successfully")
    scaler = joblib.load('scaler.pkl')
    print("Scaler Loaded successfully")
except Exception as e:
    print(f"Failed to load pickles: {e}")

print("\n--- Testing Torch Load ---")
try:
    import torch.nn as nn
    class ChurnModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(7, 64) 
            self.layer2 = nn.Linear(64, 32)
            self.layer3 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.3)
            
        def forward(self, x):
            x = self.relu(self.layer1(x))
            x = self.dropout(x)
            x = self.relu(self.layer2(x))
            x = self.sigmoid(self.layer3(x))
            return x

    dl_model = ChurnModel()
    dl_model.load_state_dict(torch.load('dl_model.pth'))
    # Use weights_only=True if prompted, but for now standard
    print("DL Model Loaded successfully")
except Exception as e:
    print(f"Failed to load Torch model: {e}")
