import os
import json
import numpy as np


def model_fn(model_dir):
    """Load model from S3."""
    model_path = os.path.join(model_dir, "heart_disease_model.npy")
    model_params = np.load(model_path, allow_pickle=True).item()
    return model_params


def predict_fn(input_data, model_params):
    """Make predictions."""
    # Normalize input
    X_norm = (input_data - model_params["feature_means"]) / model_params["feature_stds"]

    # Predict
    z = X_norm @ model_params["w"] + model_params["b"]
    prob = 1 / (1 + np.exp(-z))
    prediction = (prob >= 0.5).astype(int)

    return {"probability": prob.tolist(), "prediction": prediction.tolist()}


def input_fn(request_body, content_type):
    """Parse input."""
    if content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data["features"])
    raise ValueError(f"Unsupported content type: {content_type}")


def output_fn(prediction, accept):
    """Format output."""
    return json.dumps(prediction)
