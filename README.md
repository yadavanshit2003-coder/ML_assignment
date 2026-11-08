# Online Shoppers Purchasing Intention Prediction

## a. Problem statement

Predict whether an online shopping session ends in a purchase (`Revenue`) from the visitor's browsing behaviour and session attributes. This is a binary classification problem. Five classifiers are trained on the same dataset, compared on the same metrics, and served from saved models in a Streamlit app. The app does not retrain on uploaded test data.

## b. Dataset description

**Online Shoppers Purchasing Intention** (UCI Machine Learning Repository).

- Instances: 12,330 in the raw file. 125 exact duplicate rows were removed before training, leaving 12,205
- Features: 17 (above the 12 minimum)
- Target: `Revenue` (`True` / `1` = purchase, `False` / `0` = no purchase)
- Class balance after dedup: 1,908 purchases and 10,297 non-purchases (imbalanced)
- Feature types: numeric session stats (page counts, durations, bounce rate, exit rate, page values, special day), categorical fields (`Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`), and a boolean `Weekend`

Numeric features are scaled and categorical features are one-hot encoded inside each saved pipeline. Train/test split is 80/20, stratified, `random_state=42`, after duplicates are removed. `test_data.csv` is that held-out split (2,441 rows) and includes the target column.

## c. GitHub repository link

https://github.com/yadavanshit2003-coder/ML_assignment

Live Streamlit app: https://mlassignment-gscl5dmr4xsgm6pyprdrzf.streamlit.app/

## d. Models used

All five models are trained on the same dataset. Metrics below are on the held-out test set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8873 | 0.8998 | 0.7536 | 0.4162 | 0.5363 | 0.5055 |
| Decision Tree | 0.9021 | 0.9243 | 0.7061 | 0.6414 | 0.6722 | 0.6157 |
| KNN | 0.8742 | 0.8002 | 0.6682 | 0.3901 | 0.4926 | 0.4465 |
| Naive Bayes | 0.2773 | 0.7508 | 0.1756 | 0.9791 | 0.2978 | 0.1375 |
| Support Vector Machine | 0.8955 | 0.8738 | 0.7378 | 0.5157 | 0.6071 | 0.5607 |

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | High accuracy (0.8873) and a strong AUC (0.8998), with the best precision (0.7536). Recall is low (0.4162), so many actual purchases are missed. |
| Decision Tree | Best accuracy (0.9021), AUC (0.9243), F1 (0.6722), and MCC (0.6157). Recall (0.6414) is clearly higher than the linear models while precision stays above 0.70. |
| KNN | Accuracy stays high (0.8742) because the majority class is easy, but AUC (0.8002) and F1 (0.4926) are weaker. Neighbour voting does not separate the rare purchase class well. |
| Naive Bayes | Recall is very high (0.9791) but precision (0.1756) and accuracy (0.2773) collapse. The independence assumption does not hold on these correlated session features, so the model flags almost everything as a purchase. |
| Support Vector Machine | Close to logistic regression: accuracy 0.8955, AUC 0.8738, F1 0.6071, MCC 0.5607. Better recall than logistic regression, slightly lower precision. |
| Overall, Winner for your dataset? | Decision Tree. It leads on accuracy, AUC, F1, and MCC, which matter more than raw accuracy on this imbalanced purchase label. |

## Project structure

```text
ML_assignment/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── online_shoppers_intention.csv
├── model_metrics.csv
├── models.pkl
└── logistic_regression.ipynb
```

## Run locally

```bash
git clone https://github.com/yadavanshit2003-coder/ML_assignment
cd ML_assignment
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Upload `test_data.csv` in the app, pick a model, and the page shows metrics plus a confusion matrix from the saved model.
