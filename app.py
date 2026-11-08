import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path


from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)


# Page configuration
st.set_page_config(
    page_title="Revenue Prediction",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Online Shoppers Revenue Prediction")
st.write("Machine Learning Model Performance Dashboard")


METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
ORIGINAL_PATH = Path("online_shoppers_intention.csv")
TEST_PATH = Path("test_data.csv")




def to_binary(series):
    """Convert True/False or 1/0 values into binary labels."""
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": 1,
            "1": 1,
            "false": 0,
            "0": 0
        })
    )


    if mapped.isna().any():
        raise ValueError(
            f"Column '{series.name}' must contain only True/False or 1/0."
        )


    return mapped.astype(int)




def score_frame(models, frame):
    """Score every saved model on a labelled frame. No retraining."""
    y_true = to_binary(frame["Revenue"])
    X = frame.drop(columns=["Revenue"])
    rows = []
    matrices = {}


    for name, model in models.items():
        predictions = to_binary(pd.Series(model.predict(X), name="Revenue"))
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            scores = model.predict_proba(X)[:, classes.index(1)]
            auc = roc_auc_score(y_true, scores) if y_true.nunique() == 2 else None
        else:
            auc = None


        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_true, predictions),
            "AUC": auc,
            "Precision": precision_score(y_true, predictions, zero_division=0),
            "Recall": recall_score(y_true, predictions, zero_division=0),
            "F1 Score": f1_score(y_true, predictions, zero_division=0),
            "MCC": matthews_corrcoef(y_true, predictions),
        })
        matrices[name] = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist()


    return pd.DataFrame(rows), matrices




@st.cache_resource
def load_models():
    return joblib.load("models.pkl")




@st.cache_data
def cached_scores(path, _token):
    frame = pd.read_csv(path)
    return score_frame(load_models(), frame)




def show_metrics(frame):
    styled = frame.style.format({
        column: "{:.4f}" for column in METRIC_COLUMNS
    }, na_rep="N/A")
    st.dataframe(styled, use_container_width=True, hide_index=True)




MODEL_TOKEN = Path("models.pkl").stat().st_mtime
models = load_models()
test_metrics, test_matrices = cached_scores(
    str(TEST_PATH), (TEST_PATH.stat().st_mtime, MODEL_TOKEN)
)
original_metrics, original_matrices = cached_scores(
    str(ORIGINAL_PATH), (ORIGINAL_PATH.stat().st_mtime, MODEL_TOKEN)
)


# --------------------------------------------------
# 1. Dataset download
# --------------------------------------------------


st.header("1. Dataset")


download_col, test_col = st.columns(2)


with download_col:
    st.subheader("Original dataset")
    st.caption(f"{len(pd.read_csv(ORIGINAL_PATH)):,} rows, including duplicate sessions.")
    with open(ORIGINAL_PATH, "rb") as file:
        st.download_button(
            label="⬇️ Download Original Dataset",
            data=file,
            file_name="online_shoppers_intention.csv",
            mime="text/csv"
        )


with test_col:
    st.subheader("Held-out test set")
    st.caption(f"{len(pd.read_csv(TEST_PATH)):,} rows. 20% split after dropping duplicates.")
    with open(TEST_PATH, "rb") as file:
        st.download_button(
            label="⬇️ Download Test Set",
            data=file,
            file_name="test_data.csv",
            mime="text/csv"
        )


# --------------------------------------------------
# 2. Model performance comparison
# --------------------------------------------------


st.header("2. Model Performance Comparison")
st.caption(
    "Same saved models on both files. The test set is the fair score. "
    "The original file also contains the rows used for training."
)


left, right = st.columns(2)
with left:
    st.subheader("Test set")
    show_metrics(test_metrics)
with right:
    st.subheader("Original dataset")
    show_metrics(original_metrics)


# --------------------------------------------------
# 3. Individual model evaluation
# --------------------------------------------------


st.header("3. Individual Model Evaluation")


selected_model = st.selectbox(
    "Select a Machine Learning Model",
    test_metrics["Model"].tolist()
)




def show_model_metrics(title, row, matrix):
    st.subheader(title)
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{row['Accuracy']:.4f}")
    col2.metric("AUC Score", f"{row['AUC']:.4f}")
    col3.metric("Precision", f"{row['Precision']:.4f}")
    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{row['Recall']:.4f}")
    col5.metric("F1 Score", f"{row['F1 Score']:.4f}")
    col6.metric("MCC Score", f"{row['MCC']:.4f}")
    st.dataframe(
        pd.DataFrame(
            matrix,
            index=["Actual No Purchase", "Actual Purchase"],
            columns=["Predicted No Purchase", "Predicted Purchase"]
        ),
        use_container_width=True
    )




test_row = test_metrics[test_metrics["Model"] == selected_model].iloc[0]
original_row = original_metrics[original_metrics["Model"] == selected_model].iloc[0]


metric_left, metric_right = st.columns(2)
with metric_left:
    show_model_metrics("Test set", test_row, test_matrices[selected_model])
with metric_right:
    show_model_metrics("Original dataset", original_row, original_matrices[selected_model])


# --------------------------------------------------
# 4. Model comparison chart
# --------------------------------------------------


st.header("4. Model Comparison Chart")


metric_choice = st.selectbox(
    "Choose a metric to compare",
    METRIC_COLUMNS
)


chart_data = pd.DataFrame({
    "Test set": test_metrics.set_index("Model")[metric_choice],
    "Original dataset": original_metrics.set_index("Model")[metric_choice],
})
st.bar_chart(chart_data)


# --------------------------------------------------
# 5. Upload test dataset and make predictions
# --------------------------------------------------


st.header("5. Test Your Own Dataset")


st.write(
    "Upload a CSV file containing the customer features. "
    "You may also include the actual Revenue column to evaluate performance."
)


uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)




if uploaded_file is not None:


    try:
        test_data = pd.read_csv(uploaded_file)


        st.subheader("Uploaded Dataset Preview")
        st.dataframe(test_data.head())


        # Separate features and actual target, if available
        if "Revenue" in test_data.columns:
            y_test = to_binary(test_data["Revenue"])
            X_test = test_data.drop(columns=["Revenue"])
        else:
            y_test = None
            X_test = test_data.copy()


        # Select the saved model
        model = models[selected_model]


        # Generate predictions
        predictions = model.predict(X_test)
        predictions = to_binary(pd.Series(predictions))


        # Get prediction scores
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)


            classes = list(model.classes_)
            positive_index = classes.index(1)


            scores = probabilities[:, positive_index]
            score_label = "Purchase Probability"


        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            score_label = "Purchase Decision Score"


        else:
            scores = None
            score_label = None


        # Prepare prediction output
        result = X_test.copy()


        result["Predicted Revenue"] = np.where(
            predictions == 1, "Purchase", "No Purchase"
        )


        if scores is not None:
            result[score_label] = scores


        st.subheader("Prediction Results")
        st.dataframe(result.head(20))


        # Download predictions
        st.download_button(
            label="⬇️ Download Predictions",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="revenue_predictions.csv",
            mime="text/csv"
        )


        # Evaluate only when actual Revenue is available
        if y_test is not None:


            st.subheader("Test Dataset Performance")


            auc = (
                roc_auc_score(y_test, scores)
                if scores is not None and y_test.nunique() == 2
                else None
            )


            col1, col2, col3 = st.columns(3)


            col1.metric(
                "Accuracy",
                f"{accuracy_score(y_test, predictions):.4f}"
            )


            col2.metric(
                "Precision",
                f"{precision_score(y_test, predictions, zero_division=0):.4f}"
            )


            col3.metric(
                "Recall",
                f"{recall_score(y_test, predictions, zero_division=0):.4f}"
            )


            col4, col5, col6 = st.columns(3)


            col4.metric(
                "F1 Score",
                f"{f1_score(y_test, predictions, zero_division=0):.4f}"
            )


            col5.metric(
                "MCC",
                f"{matthews_corrcoef(y_test, predictions):.4f}"
            )


            col6.metric(
                "AUC",
                f"{auc:.4f}" if auc is not None else "N/A"
            )


            st.subheader("Confusion Matrix")


            cm = confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1]
            )


            st.dataframe(
                pd.DataFrame(
                    cm,
                    index=["Actual No Purchase", "Actual Purchase"],
                    columns=["Predicted No Purchase", "Predicted Purchase"]
                )
            )


        else:
            st.info(
                "The uploaded CSV does not contain a Revenue column. "
                "Predictions are available, but performance metrics "
                "cannot be calculated without actual labels."
            )


    except Exception as e:
        st.error(f"Could not process the uploaded dataset: {e}")


# --------------------------------------------------
# 6. Prediction information
# --------------------------------------------------


st.header("6. Customer Revenue Prediction")


st.info(
    "The selected saved model is: " + selected_model
)


st.write(
    "You can upload a test dataset above to generate predictions "
    "using this model."
)