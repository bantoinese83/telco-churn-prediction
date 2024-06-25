import pandas as pd
import os
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    precision_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from app.logger_config import configure_logger


# Configure loguru
logger = configure_logger()

# Set pandas options
pd.set_option("future.no_silent_downcasting", True)


def load_dataset(dataset_path):
    logger.info("Loading dataset from {}", dataset_path)
    return pd.read_csv(dataset_path)


def preprocess_data(churn_df):
    logger.info("Converting TotalCharges to numeric and filling NaN values")
    churn_df["TotalCharges"] = pd.to_numeric(churn_df["TotalCharges"], errors="coerce")
    churn_df["TotalCharges"] = churn_df["TotalCharges"].fillna(
        churn_df["tenure"] * churn_df["MonthlyCharges"]
    )

    logger.info("Converting SeniorCitizen to object")
    churn_df["SeniorCitizen"] = churn_df["SeniorCitizen"].astype(object)

    logger.info("Replacing 'No phone service' and 'No internet service' with 'No'")
    churn_df["MultipleLines"] = churn_df["MultipleLines"].replace(
        "No phone service", "No"
    )
    columns_to_replace = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for column in columns_to_replace:
        churn_df[column] = churn_df[column].replace("No internet service", "No")

    logger.info("Converting 'Churn' categorical variable to numeric")
    churn_df["Churn"] = churn_df["Churn"].replace({"No": 0, "Yes": 1}).astype(int)

    return churn_df


def split_data(churn_df):
    logger.info("Creating StratifiedShuffleSplit object")
    stratified_split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=64
    )

    train_index, test_index = next(stratified_split.split(churn_df, churn_df["Churn"]))

    logger.info("Creating training and test sets")
    train_set = churn_df.loc[train_index]
    test_set = churn_df.loc[test_index]

    X_train = train_set.drop("Churn", axis=1)
    y_train = train_set["Churn"].copy()

    X_test = test_set.drop("Churn", axis=1)
    y_test = test_set["Churn"].copy()

    return X_train, X_test, y_train, y_test


def save_data(churn_df, X_train, X_test, y_train, y_test):
    logger.info("Saving preprocessed data and train/test sets")
    churn_df.to_parquet("data/churn_data_regulated.parquet")
    X_train.to_pickle("data/X_train.pkl")
    X_test.to_pickle("data/X_test.pkl")
    y_train.to_pickle("data/y_train.pkl")
    y_test.to_pickle("data/y_test.pkl")


def train_and_save_model(X_train, y_train, X_test, y_test):
    # Identify categorical columns
    logger.info("Identifying categorical columns")
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    # Initialize and fit CatBoostClassifier
    logger.info("Initializing CatBoostClassifier")
    churn_model = CatBoostClassifier(verbose=False, random_state=0, scale_pos_weight=3)

    logger.info("Fitting CatBoostClassifier")
    churn_model.fit(
        X_train, y_train, cat_features=categorical_features, eval_set=(X_test, y_test)
    )

    # Save the model in the 'model' directory
    model_directory = "model"
    logger.info("Saving model to {}", model_directory)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    model_filepath = os.path.join(model_directory, "catboost_model.cbm")
    churn_model.save_model(model_filepath)
    logger.info("Model saved successfully at {}", model_filepath)

    return churn_model


def evaluate_model(churn_model, X_test, y_test):
    # Predict on a test set
    logger.info("Making predictions on the test set")
    y_pred = churn_model.predict(X_test)

    # Calculate evaluation metrics
    logger.info("Calculating evaluation metrics")
    try:
        accuracy, recall, roc_auc, precision = [
            round(metric(y_test, y_pred), 4)
            for metric in [accuracy_score, recall_score, roc_auc_score, precision_score]
        ]
        logger.info("Evaluation metrics calculated successfully")
    except Exception as e:
        logger.error("Error calculating evaluation metrics: {}", e)
        raise

    # Create a DataFrame to store results
    logger.info("Creating DataFrame to store evaluation results")
    model_results = pd.DataFrame(
        {
            "Accuracy": accuracy,
            "Recall": recall,
            "Roc_Auc": roc_auc,
            "Precision": precision,
        },
        index=["CatBoost_Model"],
    )

    # Print results
    logger.info("Model evaluation results:\n{}", model_results)


def train_model():
    dataset_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    churn_df = load_dataset(dataset_path)
    churn_df = preprocess_data(churn_df)
    X_train, X_test, y_train, y_test = split_data(churn_df)
    save_data(churn_df, X_train, X_test, y_train, y_test)
    churn_model = train_and_save_model(X_train, y_train, X_test, y_test)
    evaluate_model(churn_model, X_test, y_test)
