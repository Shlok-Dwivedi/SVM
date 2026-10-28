# Lab 8: Support Vector Machine (SVM) on AI4I 2020 Dataset

Binary classification of machine failure for predictive maintenance using Support Vector Machines (Linear and RBF Kernels) and GridSearchCV hyperparameter tuning from Scikit-learn.

## Overview

The notebook covers exploratory data analysis, class imbalance analysis, data preprocessing using `ColumnTransformer` (`StandardScaler` and `OneHotEncoder`), training Linear and RBF SVM models, hyperparameter optimization with `GridSearchCV` across regularization parameters ($C$) and kernel coefficients ($\gamma$), performance evaluation using Precision, Recall, F1-Score, Confusion Matrices, ROC/AUC Curves, and inference on unseen machine operating samples.
