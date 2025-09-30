    # main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# Define the input data structure using Pydantic
class BPM_Input(BaseModel):
    resting_hr: float
    hr_mean_24h: float

# Initialize FastAPI app
app = FastAPI()

# Load the trained model and scaler at startup
# These assets were saved after training and scaling the data
try:
    model = joblib.load('bpm_predictor_model.joblib')
    scaler = joblib.load('bpm_scaler.joblib')
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Model or scaler file not found. Make sure 'bpm_predictor_model.joblib' and 'bpm_scaler.joblib' are in the same directory.")
    model = None
    scaler = None

# Define the prediction endpoint
@app.post("/predict_bpm_risk")
def predict_bpm_risk(data: BPM_Input):
    if model is None or scaler is None:
        return {"error": "Model or scaler not loaded. Cannot make predictions."}

    # Calculate standard deviation of the two input values
    hr_std_24h = np.std([data.resting_hr, data.hr_mean_24h])
    
    # Convert input data to a NumPy array
    input_data = np.array([[data.resting_hr, data.hr_mean_24h, hr_std_24h]])

    # Apply the loaded scaler to transform the input data
    scaled_data = scaler.transform(input_data)

    # Get prediction probabilities
    # The model is trained to predict the probability of the positive class (Cardiac Event)
    probabilities = model.predict_proba(scaled_data)

    # The probability of the positive class is at index 1
    risk_score = probabilities[:, 1][0]

    # Determine human-readable prediction based on a threshold (e.g., 0.5)
    # This threshold can be adjusted based on the desired balance between precision and recall
    prediction = "High Risk" if risk_score >= 0.5 else "Low Risk"


    return {
        "prediction": prediction,
        "risk_score": float(risk_score) # Return as float for JSON serialization
    }

# Instructions on how to run the API server:
# Save this code as main.py
# Install necessary libraries: pip install fastapi uvicorn python-multipart joblib numpy scikit-learn xgboost
# Run the server from your terminal: uvicorn main:app --reload
# The API documentation will be available at http://127.0.0.1:8000/docs