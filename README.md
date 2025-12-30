## Bank Marketing Term Deposit Prediction

### Objective
Predict whether a bank client will subscribe to a term deposit using classification models.

### Dataset
UCI Machine Learning Repository – Bank Marketing Dataset (ID: 222)  
Loaded directly via `ucimlrepo` to ensure reproducibility.

### Methods
- Logistic Regression
- Random Forest
- SMOTE for class imbalance
- PCA for dimensionality reduction

### Evaluation Metrics
F1-score (minority class), ROC-AUC, PR-AUC

### Tools
Python, scikit-learn, imbalanced-learn, Google Colab

### Key Result
Random Forest with PCA and SMOTE achieved the best performance in terms of F1-score and PR-AUC.
