# Credit Card Fraud Detection — NeoFraud

An end-to-end machine learning project for detecting fraudulent credit card transactions. The project includes four individual algorithm notebooks and a Streamlit web application ("NeoFraud") that lets users upload a dataset, run multiple classifiers side by side, and compare their performance interactively.

---

## Project Structure

```
credit_card_fraud_detection/
├── credit_card_fraud_detection_app.py   # Streamlit web application
├── ADA_BOOSt_2.ipynb                    # AdaBoost classifier notebook
├── decision_tree_algorithm.ipynb        # Decision Tree classifier notebook
├── KNN_algorithm.ipynb                  # K-Nearest Neighbors notebook
├── Naive_Bayes__GaussianNB_algorithm.ipynb  # Gaussian Naive Bayes notebook
└── .gitignore
```

---

## Dataset

All notebooks and the app expect a CSV file named `creditcard.csv` (or uploaded via the app). The dataset must contain a `Class` column as the binary target label where `1` = Fraud and `0` = Genuine.

The standard dataset used for this problem is the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains 284,807 transactions with 492 fraud cases (~0.17% fraud rate), with features `V1`–`V28` (PCA-transformed), `Time`, and `Amount`.

---

## Models

| Notebook | Algorithm | Key Configuration |
|----------|-----------|-------------------|
| `Naive_Bayes__GaussianNB_algorithm.ipynb` | Gaussian Naive Bayes | StandardScaler pipeline; 80/20 stratified split |
| `KNN_algorithm.ipynb` | K-Nearest Neighbors | k=5, Minkowski distance (p=2); 75/25 split; PCA visualization |
| `ADA_BOOSt_2.ipynb` | AdaBoost | Base estimator: DecisionTree (max_depth=1); n_estimators=25; learning_rate=1.0; SMOTE oversampling; 80/20 stratified split |
| `decision_tree_algorithm.ipynb` | Decision Tree | Gini criterion; max_depth=4; class_weight={0:1, 1:3}; 70/30 stratified split |

---

## Metrics Reported

Each notebook and the app report the following metrics for both training and test sets, with a focus on the fraud class (label `1`):

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve & AUC score (AdaBoost)

---

## NeoFraud — Streamlit App

### Features

- **Upload** any CSV dataset with a `Class` column
- **Select** one or more models to run simultaneously (Naive Bayes, KNN, AdaBoost, Decision Tree)
- **Model Laboratory tab** — per-model results with training and test metrics, confusion matrices, and model-specific visualizations (PCA scatter plot for KNN, ROC curve and error plot for AdaBoost, tree visualization for Decision Tree)
- **Performance Dashboard tab** — side-by-side comparison tables (color-graded) and grouped bar chart across all selected models

### Running the App

**Install dependencies:**

```bash
pip install streamlit pandas numpy scikit-learn imbalanced-learn matplotlib seaborn plotly streamlit-extras
```

**Run:**

```bash
streamlit run credit_card_fraud_detection_app.py
```

Then open `http://localhost:8501` in your browser, upload `creditcard.csv` via the sidebar, select your models, and explore the results.

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `scikit-learn` | All ML models, preprocessing, metrics |
| `imbalanced-learn` | SMOTE oversampling (AdaBoost notebook) |
| `pandas`, `numpy` | Data handling |
| `matplotlib`, `seaborn` | Static visualizations (notebooks) |
| `plotly` | Interactive charts (Streamlit app) |
| `streamlit` | Web application framework |
| `streamlit-extras` | Stylable containers and metric cards |

---

## Notes

- The dataset is highly imbalanced (~0.17% fraud). The AdaBoost notebook uses **SMOTE** to oversample the minority class before training. The Decision Tree uses **class weighting** (`{0:1, 1:3}`) as an alternative approach.
- All models apply `StandardScaler` before fitting. The KNN notebook applies scaling manually; all others use a `sklearn.pipeline.Pipeline`.
- The app caches uploaded data with `@st.cache_data` to avoid re-loading on every interaction.
