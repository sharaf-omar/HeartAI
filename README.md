# Heart Eco-System: AI Predictive Engine

This repository contains the AI core for the *Heart Eco-System* project, a *Digitopia AI Challenge* entry. The purpose of this part is to build a highly interpretable and accurate machine learning model that can forecast 24-hour cardiac event risk based on aggregated Beats Per Minute (BPM) values alone from patients' data.

## Project Mission & AI Objective

- *Clinical Goal:* To provide an active device for patients and doctors to anticipate and manage the risk of cardiac events.

- *AI Goal:* To develop a robust, reliable, and explainable binary classification model that predicts the probability of a Cardiac_Event_Next_24h using only a few BPM-based features.
---

## The AI Model: XGBoost for Clinical Prediction

### Strategic Choice: Why XGBoost?

We made a conscious strategic choice to construct our model based on an *XGBoost Classifier* on 24-hour aggregated BPM metrics. This was selected for a number of important reasons:

1.  *State-of-the-Art Performance:* For tabular, structured data such as our feature set, XGBoost (eXtreme Gradient Boosting) is always one of the top-performing algorithms, able to recognize complex, non-linear patterns.

2.  *Efficiency Computation:* Compared to deep learning models requiring enormous computational budgets, XGBoost is fast to train as well as efficient for real-time inference, and thus ideal for a cloud-based scalable API.
3.  *Noise Robustness:* By using aggregated features (e.g., mean and std) over a duration of 24 hours, the model becomes inherently resistant to the transient noise and glitches in raw wearable sensor data.
4.  *Full Explainability:* Perhaps the greatest call for clinical adoption is trust. XGBoost models are fully explainable through the *SHAP (SHapley Additive exPlanations)* system, which allows us to communicate why the model made a specific risk estimate for a given patient.
### Model Architecture

-   *Model:* xgboost.XGBClassifier

-   *Input Features (BPM-only):*
    -   resting_hr: Patient's resting heart rate (BPM).
-   hr_mean_24h: Mean patient heart rate over the last 24 hours.
    -   hr_std_24h: Standard deviation of the patient heart rate for 24 hours, indicating variability or irregularity.
-   *Target Variable:*
    -   Cardiac_Event_Next_24h: Flag variable (event=1, no event=0).
-   *Key Parameter:*
-   scale_pos_weight: This is a critical parameter and is used in order to balance the class imbalance of the dataset with greater focus on classifying the less dominant "Event" class correctly.
---

## The Machine Learning Pipeline

Our model originates from a rigorous and systematic MLOps pipeline with reproducibility and best performance.

### 1. Data Foundation and Preparation

-   *Source Data:* Training is performed on train.csv and testing is done on test.csv.

-   *Feature Isolation:* Just the three columns for BPM were selected to strictly meet the project requirement.
-   *Data Scaling:* A scikit-learn StandardScaler was used to normalize features. The scaler was only fitted on the training set to prevent data leakage prior to being used to transform the training and test sets.
### 2. Automated Hyperparameter Optimization

We did not utilize default model parameters. A systematic and automated search was conducted to find the optimal model architecture.

-   *Framework:* keras-tuner was used to manage the search process.

-   *Methodology:* A RandomSearch methodology was used to attempt 30 distinct hyperparameter combinations.
-   *Objective:* The search was optimized to discover the highest *Area Under the Precision-Recall Curve (AUC-PR)*, with the most significant weight placed on our goal metric for imbalanced datasets.
The identified best hyperparameters were:

-   n_estimators: 800
-   max_depth: 7
-   learning_rate: ~0.023
-   subsample: 1.0
-   colsample_bytree: 0.6
-   min_child_weight: 6
-   gamma: 0.4
### 3. Final Model Testing

The last model, built with the optimal hyperparameters, was compared against the unseen test set.

-   *Principal Metric (Clinical):* The model performed with a *Recall of 98%* for the "Event" class, which indicates its excellent ability to identify patients who are actually at risk.

-   *Secondary Metric (Technical):* Precision-Recall Curve and Confusion Matrix also confirm that the model's high accuracy and robustness are justified.
---

## Explainable AI (XAI) for Clinical Trust

The output of this model must be explainable if it is to be trusted as a clinical tool. We made this possible by using the *SHAP* library.

-   *Global Explainability:* SHAP summary plot revealed the most influential feature to be hr_std_24h (heart rate variability), followed by hr_mean_24h. This is in accordance with clinical knowledge that unstable heart rate behavior is a good predictor of risk.

-   *Local Explainability:* We can generate for every single prediction a force plot showing exactly which of the factors contributed to the risk score and to what degree, providing a clear, evidence-based, no-room-for-misinterpretation explanation for doctors.
---

## Deployment & Operationalization

The AI model has been finalized, validated, and made explainable.

### 1. Saved Assets

Two crucial files have been saved with joblib:

-   bpm_predictor_model.joblib: Trained XGBoost model.
-   bpm_scaler.joblib: The trained StandardScaler object, used to preprocess new data.
### 2. Production API

A complete API script, main.py, has been implemented using the *FastAPI* framework.

-   *Endpoint:* It includes a single POST endpoint at /predict_bpm_risk.

-   *Input:* The endpoint receives a JSON object with the three BPM features: resting_hr, hr_mean_24h, and hr_std_24h.
-   *Process:* Upon receiving a request, the API preprocesses the received data with the loaded scaler, sends it to the model, and generates a risk probability.
-   *Output:* It sends a plain JSON response containing a human-readable prediction ("High Risk" or "Low Risk") and the risk score in a numeric format.
### How to Run the API

1.  Install all required libraries: pip install fastapi uvicorn python-multipart joblib numpy scikit-learn xgboost
2.  Put main.py, bpm_predictor_model.joblib, and bpm_scaler.joblib in the same folder.
3.  Launch the server in your terminal:
    bash
    uvicorn main:app --reload
    
4.  The API will be hosted on Railway.app, and interactive documentation can be accessed at [text](https://heartai-production.up.railway.app/docs)