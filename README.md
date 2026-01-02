## Bank Marketing Term Deposit Prediction

### Objective
Predict whether a bank client will subscribe to a term deposit using classification models, with a focus on handling class imbalance and supporting marketing decision-making.

### Dataset
- Source: UCI Machine Learning Repository – Bank Marketing Dataset (ID: 222)
- Access: Loaded directly via ucimlrepo for reproducibility
- Target: Term deposit subscription (yes / no)
- Challenge: Highly imbalanced outcome

### Methods
- Logistic Regression
- Random Forest
- SMOTE for class imbalance
- Principal Component Analysis (PCA) for dimensionality reduction
- Stratified 5-fold cross-validation

### Evaluation Metrics
- F1-score (minority class)
- PR-AUC
- ROC-AUC

### Tools
Python, scikit-learn, imbalanced-learn, pandas, numpy, matplotlib
(Google Colab and Local Jupyter Notebook)

### Key Result
- PCA did not improve Logistic Regression performance, indicating limited benefit for linear models in this dataset.
- PCA significantly improved Random Forest, increasing F1-score, PR-AUC, and ROC-AUC.
- Random Forest with PCA and SMOTE achieved the best overall performance and was selected as the final model.
