import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Revenue Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Online Shoppers Revenue Prediction")
st.write("Machine Learning Model Performance Dashboard")

# Load saved files
models = joblib.load("models.pkl")
metrics = pd.read_csv("model_metrics.csv")

# Model comparison
st.header("1. Model Performance Comparison")

st.dataframe(
    metrics.style.format({
        "Accuracy": "{:.4f}",
        "AUC": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1 Score": "{:.4f}",
        "MCC": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

# Select a model
st.header("2. Individual Model Evaluation")

selected_model = st.selectbox(
    "Select a Machine Learning Model",
    metrics["Model"].tolist()
)

# Display all six metrics
selected_metrics = metrics[
    metrics["Model"] == selected_model
].iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{selected_metrics['Accuracy']:.4f}")
col2.metric("AUC Score", f"{selected_metrics['AUC']:.4f}")
col3.metric("Precision", f"{selected_metrics['Precision']:.4f}")

col4, col5, col6 = st.columns(3)

col4.metric("Recall", f"{selected_metrics['Recall']:.4f}")
col5.metric("F1 Score", f"{selected_metrics['F1 Score']:.4f}")
col6.metric("MCC Score", f"{selected_metrics['MCC']:.4f}")

# Comparison chart
st.header("3. Model Comparison Chart")

chart_data = metrics.set_index("Model")[
    ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
]

st.bar_chart(chart_data)

# Prediction section
st.header("4. Customer Revenue Prediction")

st.info(
    "Select a model and use your customer input interface "
    "to generate a new prediction."
)

st.write("Selected model:", selected_model)