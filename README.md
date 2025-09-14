# Breast Cancer Classification Project 🧬

This project demonstrates how to build and evaluate different **machine
learning classification models** (Logistic Regression, SVM, Decision
Tree, Random Forest, Gradient Boosting, XGBoost) for breast cancer
diagnosis prediction using the **Breast Cancer Wisconsin Dataset**.

The pipeline includes data preprocessing, exploratory data analysis
(EDA), feature scaling, model training, hyperparameter tuning, and
performance evaluation.

------------------------------------------------------------------------

## 📌 1. Dataset Overview

The dataset contains multiple medical measurements of breast cancer
tumors, such as radius, texture, perimeter, area, compactness, and
concavity.\
The target column is **`diagnosis`**: - **M (Malignant)** = Cancerous
tumor\
- **B (Benign)** = Non-cancerous tumor

We encode this column numerically as: - `1` = Malignant\
- `0` = Benign

------------------------------------------------------------------------

## 📌 2. Data Preprocessing

1.  **Drop unnecessary columns** like `id` and `Unnamed: 32` (not useful
    for prediction).\
2.  **Label Encoding** is applied to convert `diagnosis` from
    categorical (M, B) to numeric (1, 0).\
3.  **Correlation Analysis** helps identify which features are strongly
    related to diagnosis.\
4.  **Data Visualization**:
    -   Pairplots for relationships between features\
    -   Countplots for class distribution\
    -   Heatmaps for correlation analysis

------------------------------------------------------------------------

## 📌 3. Feature Scaling

-   We apply **StandardScaler** to ensure all features have the same
    scale.\
-   This prevents features with larger numerical ranges from dominating
    the model.

------------------------------------------------------------------------

## 📌 4. Train-Test Split

-   The dataset is divided into **80% training** and **20% testing**
    sets.\
-   This ensures the model is trained on one set and evaluated on unseen
    data.

------------------------------------------------------------------------

## 📌 5. Model Training & Evaluation

We trained multiple models to compare performance: - Logistic
Regression\
- Linear SVM\
- Decision Tree\
- Random Forest\
- Gradient Boosting\
- XGBoost

### Evaluation Metrics Used:

-   **Accuracy** → Overall correctness of the model\
-   **Precision** → Of all predicted malignant cases, how many were
    actually malignant\
-   **Recall (Sensitivity)** → Of all actual malignant cases, how many
    were correctly detected\
-   **F1 Score** → Balance between Precision and Recall\
-   **Confusion Matrix** → Table showing TP, TN, FP, FN counts\
-   **ROC-AUC** → Measures the ability of the model to distinguish
    between classes

### Visualization:

-   Bar chart comparing accuracy across models\
-   Multi-metric chart (Accuracy, Precision, Recall, F1) for detailed
    comparison

------------------------------------------------------------------------

## 📌 6. Hyperparameter Tuning (Logistic Regression)

We used **GridSearchCV** with cross-validation for tuning Logistic
Regression:\
- Solvers: `liblinear`, `lbfgs`, `newton-cg`, `sag`, `saga`\
- Penalty types: `l1`, `l2`, `elasticnet`\
- Regularization strength (`C` values): \[0.01, 0.1, 1, 3, 10\]\
- Class weights: None, Balanced

Best model selection was based on **F1 score**, since it balances
precision and recall for imbalanced data.

Final tuned Logistic Regression model was evaluated on the test set.

------------------------------------------------------------------------

## 📌 7. Key Learnings

1.  Logistic Regression performed the best overall in this dataset.\
2.  Scaling features greatly improved performance of models like
    Logistic Regression and SVM.\
3.  Decision Trees without tuning tend to overfit.\
4.  Ensemble models (Random Forest, Gradient Boosting, XGBoost) provide
    stable performance.\
5.  Hyperparameter tuning significantly boosts model performance and
    reliability.

------------------------------------------------------------------------

## 📌 8. How to Run

``` bash
pip install -r requirements.txt
python main.py
```

------------------------------------------------------------------------

## 📌 9. Requirements

-   Python 3.x\
-   scikit-learn\
-   seaborn, matplotlib\
-   xgboost\
-   pandas, numpy

------------------------------------------------------------------------

## 📌 10. Results Summary

-   Logistic Regression achieved the highest F1-score and overall
    balance.\
-   ROC-AUC confirmed strong separation between Malignant and Benign
    classes.\
-   Visualizations helped interpret which features matter most.

------------------------------------------------------------------------

## 📌 11. Future Improvements

-   Apply **SMOTE or class balancing** for handling imbalance.\
-   Try **deep learning models** (Neural Networks).\
-   Deploy model as a **Streamlit web app**.

------------------------------------------------------------------------

## 📌 12. Conclusion

This project shows how different machine learning models can be compared
for breast cancer prediction. Logistic Regression, with proper scaling
and tuning, achieved the best results while being simple and
interpretable.
