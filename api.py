from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from feature_engineering import smiles_to_maccs, amino_acid_composition

app = FastAPI()

# Load model safely
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# Define request body properly
class InputData(BaseModel):
    smiles: str
    sequence: str


@app.get("/")
def home():
    return {"message": "DTI API is running 🚀"}


@app.post("/predict")
def predict(data: InputData):

    # Feature extraction
    maccs = np.array(smiles_to_maccs(data.smiles))
    aac = np.array(amino_acid_composition(data.sequence))

    # Combine features
    X = np.concatenate([maccs, aac]).reshape(1, -1)

    # Scale
    X = scaler.transform(X)

    # Predict
    pred = model.predict(X)[0]

    return {
        "interaction": int(pred)
    }