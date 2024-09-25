import pickle
import pandas as pd
import numpy as np
import os
import json

# Load the model from the provided S3 bucket
def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'Loan_Status.pkl')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Preprocess incoming requests, expecting JSON data
def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return pd.DataFrame([data])  # Assuming the incoming data is a single record as JSON
    raise ValueError(f"Unsupported content type: {request_content_type}")

# Prediction logic
def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    return prediction

# Output the prediction in JSON format
def output_fn(prediction, content_type):
    if content_type == 'application/json':
        return json.dumps({'predictions': prediction.tolist()})
    raise ValueError(f"Unsupported content type: {content_type}")
