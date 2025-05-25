from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Drug Effectiveness API")

# Load model and data once at startup
model = joblib.load("drug_effectiveness_model.pkl")
df = pd.read_csv("drug_clean.csv")
drug_data = df.groupby("Drug").mean(numeric_only=True).reset_index()

class PredictRequest(BaseModel):
    drug: str

class PredictResponse(BaseModel):
    drug: str
    prediction: float

@app.post("/predict", response_model=PredictResponse)
def predict_effectiveness(request: PredictRequest):
    drug = request.drug
    row = drug_data[drug_data["Drug"] == drug]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Drug '{drug}' not found")
    
    features = row[["Price", "Reviews", "Satisfaction", "EaseOfUse"]].values.reshape(1, -1)
    prediction = model.predict(features)[0]
    
    return PredictResponse(drug=drug, prediction=round(prediction, 2))
