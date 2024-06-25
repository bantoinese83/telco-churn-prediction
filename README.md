# Telco Customer Churn Prediction

This project focuses on predicting customer churn for a telecommunications company using a machine learning model built with the CatBoost library. The project includes several components such as data preprocessing, model training, prediction, and a web application for visualization and user interaction.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Streamlit Application](#streamlit-application)
- [Data Preprocessing and Training](#data-preprocessing-and-training)


## Project Overview

The goal of this project is to build a model to predict whether a customer will churn (leave the service) based on various features. The project involves the following steps:
1. Data loading and preprocessing
2. Model training using CatBoost
3. Model evaluation
4. Prediction using a FastAPI-based API
5. Visualization and user interaction using Streamlit

## Directory Structure

The project directory structure is as follows:

```
.
├── app
│   ├── __init__.py
│   ├── logger_config.py
│   ├── fast_api.py
│   ├── train_model.py
│   └── predict.py
├── assets
│   └── logo.png
├── data
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── churn_data_regulated.parquet
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   └── y_test.pkl
├── model
│   └── catboost_model.cbm
├── logs
│   └── app_log_{time}.log
├── streamlit_app.py
├── requirements.txt
├── Dockerfile
├── .gitignore
├── main.py
└── README.md
```

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/bantoinese83/telco-churn-prediction.git
    cd telco-churn-prediction
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### API Endpoints

1. Start the FastAPI server:
    ```sh
    python main.py --mode fastapi --verbose --logfile churn_app.log
    ```

2. Access the API documentation at [http://127.0.0.1:5000/docs](http://127.0.0.1:5000/docs).

3. Use the `/predict/` endpoint to get churn prediction.

### Streamlit Application

1. Start the Streamlit application:
    ```sh
    streamlit run streamlit_app.py
    ```
<img src="assets/screenshot.png" alt="Logo" width="2557"/>


## Data Preprocessing and Training

The data preprocessing and training scripts are located in the `app/train_model.py` file. This script includes data loading, preprocessing, and model training steps. The main steps involved are:

1. Loading the dataset.
2. Data cleaning and preprocessing.
3. Splitting the dataset into training and testing sets.
4. Training the CatBoost model.
5. Saving the trained model for future predictions.

To train the model with default logging:
```sh
python main.py --mode train
```



