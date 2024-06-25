import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException
from catboost import CatBoostClassifier
from pydantic import BaseModel, Field

from app.logger_config import configure_logger

logger = configure_logger()
# Path of the trained model
MODEL_PATH = "mnt/model/catboost_model.cbm"


# Function to load the trained model
def load_model():
    try:
        loaded_model = CatBoostClassifier()
        loaded_model.load_model(MODEL_PATH)
        return loaded_model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


# Define a Pydantic model for data validation
class ChurnData(BaseModel):
    customerID: str = Field(..., example="6464-UIAEA")
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="Yes")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="Yes")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=70.35)
    TotalCharges: float = Field(..., example=1397.47)


# Function to predict churn probability from data in dictionary format
def get_churn_probability(data: ChurnData, trained_model: CatBoostClassifier) -> float:
    try:
        # Convert incoming data into a DataFrame
        dataframe = pd.DataFrame([data.model_dump()])
        # Make the prediction
        churn_probability = trained_model.predict_proba(dataframe)[0][1]
        return churn_probability
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=400, detail="Error in prediction")


# Load the model
model = load_model()

# Create the FastAPI application
app = FastAPI(title="Churn Prediction API", version="1.0")


@app.get("/")
def index():
    logger.info("Accessed index endpoint")
    return {"message": "CHURN Prediction API"}


# Define the API endpoint
@app.post("/predict/")
def predict_churn(data: ChurnData):
    logger.info("Received data for prediction")
    # Load the loaded_model
    loaded_model = load_model()
    if loaded_model is None:
        raise HTTPException(status_code=500, detail="Error loading loaded_model")
    # Get the prediction
    churn_probability = get_churn_probability(data, trained_model=loaded_model)
    # Return the prediction
    return {"Churn Probability": churn_probability}


def run_fast_api():
    uvicorn.run("app.fast_api:app", host="127.0.0.1", port=5000)
