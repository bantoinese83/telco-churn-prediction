import os

import pandas as pd
from catboost import CatBoostClassifier
from app.logger_config import configure_logger

logger = configure_logger()


# Constants
MODEL_PATH = "mnt/model/catboost_model.cbm"
INPUT_PROMPTS = {
    "customerID": "Customer ID: ",
    "gender": "Gender (Male/Female): ",
    "SeniorCitizen": "Senior Citizen (0/1): ",
    "Partner": "Partner (Yes/No): ",
    "Dependents": "Dependents (Yes/No): ",
    "tenure": "Tenure (months): ",
    "PhoneService": "Phone Service (Yes/No): ",
    "MultipleLines": "Multiple Lines (Yes/No): ",
    "InternetService": "Internet Service (DSL/Fiber optic/No): ",
    "onlineSecurity": "Online Security (Yes/No): ",
    "OnlineBackup": "Online Backup (Yes/No): ",
    "DeviceProtection": "Device Protection (Yes/No): ",
    "TechSupport": "Tech Support (Yes/No): ",
    "StreamingTV": "Streaming TV (Yes/No): ",
    "StreamingMovies": "Streaming Movies (Yes/No): ",
    "Contract": "Contract (Month-to-month/One year/Two year): ",
    "Paperless_Billing": "Paperless Billing (Yes/No): ",
    "PaymentMethod": "Payment Method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card ("
    "automatic)): ",
    "MonthlyCharges": "Monthly Charges: ",
    "TotalCharges": "Total Charges: ",
}
EXPECTED_TYPES = {
    "customerID": str,
    "gender": str,
    "SeniorCitizen": int,
    "Partner": str,
    "Dependents": str,
    "tenure": int,
    "PhoneService": str,
    "MultipleLines": str,
    "InternetService": str,
    "onlineSecurity": str,
    "OnlineBackup": str,
    "DeviceProtection": str,
    "TechSupport": str,
    "StreamingTV": str,
    "StreamingMovies": str,
    "Contract": str,
    "Paperless_Billing": str,
    "PaymentMethod": str,
    "MonthlyCharges": float,
    "TotalCharges": float,
}


# Function to load the trained model
def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            loaded_model = CatBoostClassifier()
            loaded_model.load_model(MODEL_PATH)
            return loaded_model
        else:
            logger.warning(f"Model file does not exist: {MODEL_PATH}")
            return None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


# Function to predict churn probability
def predict_churn(user_input, model):
    try:
        user_data = pd.DataFrame([user_input])
        prediction = model.predict_proba(user_data)[:, 1][0]
        return {"Churn Probability": float(prediction)}
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return {"error": str(e)}


# Function to validate user input
def validate_input(input_value, expected_type):
    if not isinstance(input_value, expected_type):
        raise ValueError(f"Expected {expected_type}, but got {type(input_value)}")
    return input_value


# Function to collect user input
def collect_user_input():
    logger.info("Please enter the following information:")
    customer_data = {}
    try:
        for key, prompt in INPUT_PROMPTS.items():
            input_value = input(prompt).strip()
            if expected_type := EXPECTED_TYPES.get(key):
                customer_data[key] = validate_input(
                    expected_type(input_value), expected_type
                )
        return customer_data
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return None


# Function to display the result
def display_result(result):
    if "error" in result:
        logger.error(f"Error: {result['error']}")
    else:
        churn_probability = result["Churn Probability"]
        formatted_churn_probability = f"{churn_probability:.2%}"
        logger.info(f"Churn Probability: {formatted_churn_probability}")


def predict():
    # Load the model
    model = load_model()
    if model is None:
        return

    # Collect user input
    user_input = collect_user_input()
    if user_input is None:
        return

    # Get the prediction
    result = predict_churn(user_input, model)

    # Display the result
    display_result(result)


if __name__ == "__main__":
    predict()
