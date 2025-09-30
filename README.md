#HeartAI - BPM risk predictor app

A FastAPI-based backend service that predicts **heart risk** using resting heart rate and 24-hour mean heart rate.

The API serves a pre-trained **scikit-learn** stored as `.joblib` files.

##Features

/predict_bpm_risk: a POST endpoint that takes in resting Heart Rate and Heart Rate Mean and gets rate prediction

Deployed on railway.app