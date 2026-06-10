import joblib

def save(model, scaler):
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")