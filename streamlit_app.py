import base64
import io

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import shap
import streamlit as st
from PIL import Image
from catboost import CatBoostClassifier

# Paths to the trained model and data
MODEL_PATH = "mnt/model/catboost_model.cbm"
DATA_PATH = "mnt/data/churn_data_regulated.parquet"
LOGO_PATH = "assets/logo.png"

# Open the image file and create a copy in memory
with open(LOGO_PATH, "rb") as f:
    image = Image.open(f).copy()

# Resize the image
image = image.resize((200, 200))

# Convert the image to base64
buffered = io.BytesIO()
image.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Load the logo with a specified width
st.sidebar.markdown(
    f'<p style="text-align: center;"><img src="data:image/png;base64,{img_str}" width="200"></p>',
    unsafe_allow_html=True,
)

# Initialize user_data as an empty dictionary at the module level
user_data = {}


# Function to load the trained model
@st.cache_data
def load_model():
    try:
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"🚨 Error loading model: {e}")
        return None


# Function to load data
@st.cache_data
def load_data():
    try:
        churn_df = pd.read_parquet(DATA_PATH)
        return churn_df
    except Exception as e:
        st.error(f"🚨 Error loading data: {e}")
        return None


# Function to get feature names from loaded data
def get_feature_names(data):
    return list(data.columns)[:-1]


# Function to predict using the loaded model
def predict(model, data):
    try:
        prediction = model.predict(data)
        return (
            "🔮 Predicted churn: Yes" if prediction[0] == 1 else "🔮 Predicted churn: No"
        )
    except Exception as e:
        st.error(f"🚨 Error making prediction: {e}")
        return None


# Function to calculate SHAP values
def calculate_shap_values(model, data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    return shap_values


# Function to plot SHAP values
def plot_shap_values(shap_values, data):
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, data, show=False)
    st.pyplot(fig)
    plt.close(fig)


# Main function to build the Streamlit app
def main():
    global DATA_PATH, MODEL_PATH, user_data
    st.title("📞 Telco Customer Churn Prediction with CatBoost 😺")
    st.subheader("🔍 Predict customer churn based on their information. 📊")

    # Load data and model
    churn_df = load_data()
    model = load_model()

    if churn_df is not None and model is not None:
        # Display data overview and summary statistics
        with st.expander("📋 Data Overview"):
            st.dataframe(churn_df.head())

        with st.expander("📈 Summary Statistics"):
            st.write(churn_df.describe())

        # Radio button for chart selection
        chart_option = st.sidebar.radio(
            "📊 Select a Chart",
            (
                "📊 Feature Distributions",
                "📉 Correlation Matrix",
                "📊 Churn Distribution",
                "📉 SHAP Summary Plot",
            ),
        )

        if chart_option == "📊 Feature Distributions":
            with st.expander("📊 Feature Distributions"):
                for feature in get_feature_names(churn_df):
                    fig = px.histogram(
                        churn_df, x=feature, title=f"Distribution of {feature}"
                    )
                    st.plotly_chart(fig)

        elif chart_option == "📉 Correlation Matrix":
            with st.expander("📉 Correlation Matrix"):
                numeric_columns = churn_df.select_dtypes(include=["float64", "int64"])
                corr = numeric_columns.corr()
                fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
                st.plotly_chart(fig)

        elif chart_option == "📊 Churn Distribution":
            with st.expander("📊 Churn Distribution"):
                if "Churn" in churn_df.columns:
                    churn_counts = churn_df["Churn"].value_counts()
                    churn_mapping = {0: "No", 1: "Yes"}
                    churn_counts.index = churn_counts.index.map(churn_mapping)
                    fig = px.pie(
                        values=churn_counts,
                        names=churn_counts.index,
                        title="Churn Distribution",
                    )
                    st.plotly_chart(fig)
                else:
                    st.write("Churn column not found in data.")

        elif chart_option == "📉 SHAP Summary Plot":
            with st.expander("📉 SHAP Summary Plot"):
                shap_values = calculate_shap_values(
                    model, churn_df.drop("Churn", axis=1)
                )
                fig, ax = plt.subplots()
                shap.summary_plot(
                    shap_values,
                    churn_df.drop("Churn", axis=1),
                    plot_type="bar",
                    show=False,
                )
                st.pyplot(fig)
                plt.close(fig)

        # Dropdown to select a customerID or search for a customer by inputting their customerID
        st.subheader(
            "🔎 Select a Customer",
            help="Select a customer to view their data and predict churn.",
        )
        customerID = st.selectbox(
            "🔎 Type or select a customerID",
            churn_df["customerID"].values,
            help="You can type a customerID or select one from the dropdown.",
        )

        # Display the data for the selected customerID
        st.subheader("📋 Data for Selected Customer")
        st.write(churn_df[churn_df["customerID"] == customerID])

        # Get the data for the selected customerID
        user_data = churn_df[churn_df["customerID"] == customerID].iloc[0].drop("Churn")

    # Sidebar for optional user input
    st.sidebar.subheader(
        "📝 User Input",
        help="You can modify the values here to see how they affect the churn prediction.",
    )
    user_input = {
        feature: st.sidebar.text_input(
            f"{feature}", user_data[feature], help=f"Current value for {feature}"
        )
        for feature in get_feature_names(churn_df)
    }

    if st.button(
        "🔮 Predict Churn",
        help="Click this button to predict churn based on the current user input.",
    ):
        user_input_df = pd.DataFrame([user_input])
        prediction = predict(model, user_input_df)
        if prediction is not None:
            st.success(f"🎉 {prediction} 🎉")

    # Add a feedback section
    st.sidebar.subheader("💬 Feedback")
    st.sidebar.text_area("Please provide your feedback here:")
    if st.sidebar.button("🗳️ Submit Feedback"):
        st.sidebar.success("✅ Thank you for your feedback!")

    # Add a disclaimer
    st.sidebar.markdown(
        """
        **📢 Disclaimer:** This app is for demonstration purposes only and is not intended for production use. 
        The predictions made by the model may not be accurate. 
        Please do not provide any sensitive or personal information in the feedback section.
        """
    )

    # Add a footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with ❤️ by [Base83](https://github.com/bantoinese83)")


if __name__ == "__main__":
    main()
