# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# For handling imbalanced data
from imblearn.over_sampling import SMOTE

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading and Preparation
# Load data with the first column as index
labels = pd.read_csv(r'C:\Users\danantha\Downloads\590_Master Project\cancer subype classification\cancer subype classification\labels.csv', index_col=0)


# Merge data and labels
data['Class'] = labels['Class']

# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Remove features with zero variance
selector = VarianceThreshold()
X_var = selector.fit_transform(X)
features = X.columns[selector.get_support(indices=True)]
X = X[features]

# Log Transformation and Standardization
X_log = np.log1p(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# Feature Selection using Mutual Information
mi = mutual_info_classif(X_scaled, y, random_state=42)
mi_scores = pd.Series(mi, index=features).sort_values(ascending=False)
num_features = 200  # Adjust as needed
top_features = mi_scores.index[:num_features]
X_selected = X_scaled[:, mi_scores.index.get_indexer(top_features)]

# Split data into training and testing sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

# Address class imbalance with SMOTE
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train_full, y_train_full)

# Encode labels
le = LabelEncoder()
y_train_resampled_encoded = le.fit_transform(y_train_resampled)
y_test_encoded = le.transform(y_test)

# Model Building and Hyperparameter Tuning
# Logistic Regression
param_grid_logreg = {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs'], 'multi_class': ['multinomial']}
logreg = LogisticRegression(random_state=42)
grid_search_logreg = GridSearchCV(logreg, param_grid_logreg, cv=3, scoring='accuracy', n_jobs=-1)
grid_search_logreg.fit(X_train_resampled, y_train_resampled_encoded)
best_logreg = grid_search_logreg.best_estimator_

# Random Forest
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train_resampled, y_train_resampled)
best_rf = grid_search_rf.best_estimator_

# Support Vector Machine (SVM)
param_grid_svm = {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}
svm = SVC(probability=True, random_state=42)
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=3, scoring='accuracy', n_jobs=-1)
grid_search_svm.fit(X_train_resampled, y_train_resampled)
best_svm = grid_search_svm.best_estimator_

# XGBoost
param_grid_xgb = {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.01, 0.1]}
xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3, scoring='accuracy', n_jobs=-1)
grid_search_xgb.fit(X_train_resampled, y_train_resampled_encoded)
best_xgb = grid_search_xgb.best_estimator_

# Artificial Neural Network (ANN)
ann_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
ann_model.fit(X_train_resampled, y_train_resampled)

# Naïve Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_resampled, y_train_resampled)

# Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression
cv_scores_logreg = cross_val_score(best_logreg, X_train_resampled, y_train_resampled_encoded, cv=skf, scoring='accuracy')
print(f'Logistic Regression CV Mean Accuracy: {cv_scores_logreg.mean():.4f}')

# Random Forest
cv_scores_rf = cross_val_score(best_rf, X_train_resampled, y_train_resampled, cv=skf, scoring='accuracy')
print(f'Random Forest CV Mean Accuracy: {cv_scores_rf.mean():.4f}')

# SVM
cv_scores_svm = cross_val_score(best_svm, X_train_resampled, y_train_resampled, cv=skf, scoring='accuracy')
print(f'SVM CV Mean Accuracy: {cv_scores_svm.mean():.4f}')

# XGBoost
cv_scores_xgb = cross_val_score(best_xgb, X_train_resampled, y_train_resampled_encoded, cv=skf, scoring='accuracy')
print(f'XGBoost CV Mean Accuracy: {cv_scores_xgb.mean():.4f}')

# ANN
cv_scores_ann = cross_val_score(ann_model, X_train_resampled, y_train_resampled, cv=skf, scoring='accuracy')
print(f'ANN CV Mean Accuracy: {cv_scores_ann.mean():.4f}')

# Naïve Bayes
cv_scores_nb = cross_val_score(nb_model, X_train_resampled, y_train_resampled, cv=skf, scoring='accuracy')
print(f'Naïve Bayes CV Mean Accuracy: {cv_scores_nb.mean():.4f}')

# Compile Cross-Validation Results
cv_results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'SVM', 'XGBoost', 'ANN', 'Naïve Bayes'],
    'Mean CV Accuracy': [
        cv_scores_logreg.mean(), cv_scores_rf.mean(), cv_scores_svm.mean(),
        cv_scores_xgb.mean(), cv_scores_ann.mean(), cv_scores_nb.mean()
    ]
})

print('\nCross-Validation Results:')
print(cv_results)

# Visualization of Cross-Validation Results
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Mean CV Accuracy', data=cv_results, palette='viridis')
plt.title('Cross-Validation Accuracy Comparison')
plt.ylabel('Mean CV Accuracy')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
