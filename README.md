# Online Shoppers Purchasing Intention Prediction

## Project Overview

This project uses machine learning classification algorithms to predict whether an online visitor will generate revenue (make a purchase) based on their browsing behaviour and session information.

The project compares multiple classification models and presents their performance through an interactive web application built using **Streamlit**.

## Dataset

The project uses the **Online Shoppers Purchasing Intention Dataset**.

- Number of records: 12,330
- Target variable: `Revenue`
- `Revenue = 1`: Purchase made
- `Revenue = 0`: No purchase made

The dataset contains numerical and categorical features describing visitors' browsing sessions.

## Machine Learning Models

The following five classification algorithms are implemented and compared:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## Model Evaluation

The models are evaluated using the following performance metrics:

| Metric | Description |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Proportion of predicted purchases that are actual purchases |
| Recall | Proportion of actual purchases correctly identified |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Measures the model's ability to distinguish between the two classes |
| Matthews Correlation Coefficient (MCC) | Measures the quality of binary classifications, accounting for all four confusion-matrix outcomes |

Since the dataset has an imbalanced target variable, multiple evaluation metrics are considered rather than relying only on accuracy.

## Project Structure

```text
ML-assignment-2/
│
├── app.py                  # Streamlit web application
├── models.pkl              # Saved trained machine learning models
├── model_metrics.csv       # Evaluation metrics for all models
├── input_schema.json       # Feature information for prediction inputs (if used)
├── requirements.txt        # Project dependencies
├── logistic_regression.ipynb  # Model development notebook
└── README.md               # Project documentation
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the project directory

```bash
cd ML-assignment-2
```

### 3. Create a virtual environment (optional)

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## Streamlit Application

The Streamlit application provides an interface to:

- View and compare the performance of all five machine learning models.
- Examine evaluation metrics such as accuracy, precision, recall, F1-score, ROC-AUC, and MCC.
- Explore model performance through visualizations.
- Make predictions using the trained models, if prediction inputs are enabled in the application.

## Deployment

The application can be deployed using **Streamlit Community Cloud**.

1. Push the project files to a GitHub repository.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in using GitHub.
4. Select the project repository.
5. Choose `app.py` as the main file.
6. Click **Deploy**.

## Author

**Your Name**

Machine Learning Assignment

---

*This project was developed as part of a Machine Learning assignment to study, implement, and compare different classification algorithms.*