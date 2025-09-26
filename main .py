# main.py - FastAPI script for heartbeat arrhythmia classification API

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import tensorflow as tf
import numpy as np
import os

# Define the path to the saved model
MODEL_PATH = 'arrhythmia_model.h5'

# Load the trained model
# Use a custom_objects dictionary if you have custom layers or functions
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set model to None if loading fails

# Define class mapping (must match the mapping used during training)
# In a real application, this might be loaded from a configuration file
CLASS_MAPPING = {
    0: 'Normal Beat',
    1: 'Supraventricular Ectopic Beat',
    2: 'Ventricular Ectopic Beat',
    3: 'Fusion Beat',
    4: 'Unknown Beat'
}

# Define the input data schema using Pydantic
class HeartbeatInput(BaseModel):
    heartbeat: List[float] = Field(..., description="A list of 187 floating-point values representing a single heartbeat signal.")

# Initialize the FastAPI app
app = FastAPI(
    title="Heartbeat Arrhythmia Classification API",
    description="API for classifying heartbeat signals into different arrhythmia categories using a 1D CNN.",
    version="1.0.0",
)

# Define the prediction endpoint
@app.post("/predict")
async def predict_arrhythmia(input_data: HeartbeatInput):
    if model is None:
        return {"error": "Model not loaded. Cannot make predictions."}

    # Preprocessing: Convert the input list to a NumPy array and reshape
    try:
        heartbeat_array = np.array(input_data.heartbeat, dtype=np.float32)
        # Ensure the input has 187 features
        if heartbeat_array.shape[0] != 187:
            return {"error": f"Invalid input length. Expected 187 features, but received {heartbeat_array.shape[0]}."}
        # Reshape to (1, 187, 1) for the model
        heartbeat_reshaped = heartbeat_array.reshape(1, 187, 1)
    except Exception as e:
        return {"error": f"Error during input preprocessing: {e}"}

    # Prediction
    try:
        predictions = model.predict(heartbeat_reshaped)
        # Get the predicted class index
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        # Map the index to the class name
        predicted_class_name = CLASS_MAPPING.get(predicted_class_index, "Unknown Class")
    except Exception as e:
        return {"error": f"Error during model prediction: {e}"}


    # Return the prediction
    return {"predicted_arrhythmia": predicted_class_name}

# Optional: Add a root endpoint for basic health check
@app.get("/")
async def read_root():
    return {"message": "Heartbeat Arrhythmia Classification API is running."}
