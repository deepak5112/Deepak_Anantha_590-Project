# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest, chi2, RFE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# For handling imbalanced data
from imblearn.over_sampling import SMOTE, ADASYN

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading and Preparation
# Load data with the first column as index
data = pd.read_csv('data.csv', index_col=0)
labels = pd.read_csv('labels.csv', index_col=0)

# Merge data and labels
data['Class'] = labels['Class']

# Display the shape of the data
print(f'Data shape: {data.shape}')

# 2. Exploratory Data Analysis (EDA)
# 2.1 Visualizing Class Distribution
# Count of each class
class_counts = data['Class'].value_counts()
print('Class Counts:')
print(class_counts)

# Plot class distribution
plt.figure(figsize=(8, 8))
class_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('Class Distribution')
plt.ylabel('')
plt.show()

# 2.2 Checking for Missing Values
# Check for missing values
print('Missing values in each column:')
print(data.isnull().sum())

# Total missing values
total_missing = data.isnull().sum().sum()
print(f'Total missing values: {total_missing}')

# 2.3 Statistical Summary
# Statistical summary of the data
print('Statistical Summary:')
print(data.describe())

# 2.4 Gene Expression Distribution (Updated)
# Plot distribution of a sample of genes using violin plots

# Sample every 1000th gene to reduce the number of plots
sample_genes = data.columns[::1000]
sample_genes = sample_genes.drop('Class', errors='ignore')

# Create a melted DataFrame for seaborn
melted_data = data[sample_genes].melt(var_name='Gene', value_name='Expression')

plt.figure(figsize=(14, 8))
sns.violinplot(x='Gene', y='Expression', data=melted_data, palette='Set3', inner='quartile')
plt.title('Distribution of Gene Expressions (Sampled Genes)')
plt.xlabel('Genes')
plt.ylabel('Expression Level')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3. Data Preprocessing
# 3.1 Removing Constant Features
# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']

print(f'Features before removing constant features: {X.shape[1]}')

# Remove features with zero variance
selector = VarianceThreshold()
X_var = selector.fit_transform(X)

# Get the list of features that are not constant
features = X.columns[selector.get_support(indices=True)]

# Update X with selected features
X = X[features]

print(f'Features after removing constant features: {X.shape[1]}')

# 3.2 Log Transformation
# Apply log transformation to reduce skewness
X_log = np.log1p(X)

# 3.3 Standardization
# Standardize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# 4. Feature Selection
# 4.1 Mutual Information
# Calculate mutual information
mi = mutual_info_classif(X_scaled, y, random_state=42)

# Create a series with feature names and MI scores
mi_scores = pd.Series(mi, index=features)
mi_scores = mi_scores.sort_values(ascending=False)

# Number of features to select
num_features = 200  # Adjust as needed

# Select top features based on mutual information
top_features = mi_scores.index[:num_features]
X_selected = X_scaled[:, mi_scores.index.get_indexer(top_features)]
print(f'Features selected (Top {num_features} MI features): {X_selected.shape[1]}')

# 4.2 Principal Component Analysis (PCA)
# Apply PCA to reduce dimensionality for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_selected)

# Plot PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Gene Expression Data')
plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 4.3 t-Distributed Stochastic Neighbor Embedding (t-SNE)
# Apply t-SNE for visualization
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
X_tsne = tsne.fit_transform(X_selected)

# Plot t-SNE
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='viridis')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE of Gene Expression Data')
plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 5. Addressing Class Imbalance (Updated)
# Split data into training and testing sets before applying SMOTE
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE only on the training data
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train_full, y_train_full)

# Display new class counts
resampled_counts = pd.Series(y_train_resampled).value_counts()
print('Resampled Training Class Counts:')
print(resampled_counts)

# 6. Model Building with Hyperparameter Tuning
# 6.1 Encode labels for models that require numerical labels
le = LabelEncoder()
y_train_resampled_encoded = le.fit_transform(y_train_resampled)
y_test_encoded = le.transform(y_test)

# 6.2 Hyperparameter Tuning with GridSearchCV
# 6.2.1 Logistic Regression
param_grid_logreg = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'saga', 'newton-cg'],
    'multi_class': ['multinomial'],
    'max_iter': [1000, 2000, 5000]
}

logreg = LogisticRegression(random_state=42)

grid_search_logreg = GridSearchCV(estimator=logreg, param_grid=param_grid_logreg,
                                  cv=3, scoring='accuracy', n_jobs=-1)

grid_search_logreg.fit(X_train_resampled, y_train_resampled_encoded)

print("Best parameters for Logistic Regression:", grid_search_logreg.best_params_)

best_logreg = grid_search_logreg.best_estimator_

# Predictions
y_pred_logreg_encoded = best_logreg.predict(X_test)
y_pred_logreg = le.inverse_transform(y_pred_logreg_encoded)

# 6.2.2 Random Forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)

grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf,
                              cv=3, scoring='accuracy', n_jobs=-1)

grid_search_rf.fit(X_train_resampled, y_train_resampled)

print("Best parameters for Random Forest:", grid_search_rf.best_params_)

best_rf = grid_search_rf.best_estimator_

# Predictions
y_pred_rf = best_rf.predict(X_test)

# 6.2.3 Support Vector Machine (SVM)
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

svm = SVC(probability=True, random_state=42)

grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm,
                               cv=3, scoring='accuracy', n_jobs=-1)

grid_search_svm.fit(X_train_resampled, y_train_resampled)

print("Best parameters for SVM:", grid_search_svm.best_params_)

best_svm = grid_search_svm.best_estimator_

# Predictions
y_pred_svm = best_svm.predict(X_test)

# 6.2.4 XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}

# Remove 'use_label_encoder' and adjust parameters
xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)

grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb,
                               cv=3, scoring='accuracy', n_jobs=-1)

grid_search_xgb.fit(X_train_resampled, y_train_resampled_encoded)

print("Best parameters for XGBoost:", grid_search_xgb.best_params_)

best_xgb = grid_search_xgb.best_estimator_

# Predictions
y_pred_xgb_encoded = best_xgb.predict(X_test)
y_pred_xgb = le.inverse_transform(y_pred_xgb_encoded)

# 6.3 Artificial Neural Network (ANN)
ann_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

ann_model.fit(X_train_resampled, y_train_resampled)

y_pred_ann = ann_model.predict(X_test)

# 6.4 Naïve Bayes Classifier
nb_model = GaussianNB()

nb_model.fit(X_train_resampled, y_train_resampled)

y_pred_nb = nb_model.predict(X_test)

# 7. Model Evaluation and Comparison
# 7.1 Evaluation Metrics
def calculate_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f'\n{model_name} Model Performance:')
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred))
    return acc, prec, rec, f1

# 7.2 Evaluating Each Model
# Evaluate Logistic Regression
acc_logreg, prec_logreg, rec_logreg, f1_logreg = calculate_metrics(y_test, y_pred_logreg, 'Logistic Regression')

# Evaluate Random Forest
acc_rf, prec_rf, rec_rf, f1_rf = calculate_metrics(y_test, y_pred_rf, 'Random Forest')

# Evaluate SVM
acc_svm, prec_svm, rec_svm, f1_svm = calculate_metrics(y_test, y_pred_svm, 'SVM')

# Evaluate XGBoost
acc_xgb, prec_xgb, rec_xgb, f1_xgb = calculate_metrics(y_test, y_pred_xgb, 'XGBoost')

# Evaluate ANN
acc_ann, prec_ann, rec_ann, f1_ann = calculate_metrics(y_test, y_pred_ann, 'ANN')

# Evaluate Naïve Bayes
acc_nb, prec_nb, rec_nb, f1_nb = calculate_metrics(y_test, y_pred_nb, 'Naïve Bayes')

# 7.3 Model Comparison
models = ['Logistic Regression', 'Random Forest', 'SVM', 'XGBoost', 'ANN', 'Naïve Bayes']
accuracy = [acc_logreg, acc_rf, acc_svm, acc_xgb, acc_ann, acc_nb]
precision = [prec_logreg, prec_rf, prec_svm, prec_xgb, prec_ann, prec_nb]
recall = [rec_logreg, rec_rf, rec_svm, rec_xgb, rec_ann, rec_nb]
f1 = [f1_logreg, f1_rf, f1_svm, f1_xgb, f1_ann, f1_nb]

comparison_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
})

print('\nModel Comparison:')
print(comparison_df)

# 7.4 Visualization of Model Performance
comparison_df.set_index('Model', inplace=True)

comparison_df.plot(kind='bar', figsize=(12, 8))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 7.5 Confusion Matrix for Logistic Regression
from sklearn.metrics import ConfusionMatrixDisplay

plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_logreg, display_labels=le.classes_, cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 8. Biological Interpretation
# 8.1 Investigating Top Features
# Get absolute values of coefficients from the best logistic regression model
coefficients = np.abs(best_logreg.coef_)
mean_coefficients = np.mean(coefficients, axis=0)

# Create a DataFrame
feature_importance = pd.DataFrame({'Feature': top_features[:len(mean_coefficients)], 'Importance': mean_coefficients})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Display top 10 features
top_features_importance = feature_importance.head(10)
print("Top 10 Features from Logistic Regression:")
print(top_features_importance)

# 8.2 Mapping Features to Gene Names
# Assuming the features correspond to gene names
for gene in top_features_importance['Feature']:
    print(f"Investigating gene: {gene}")
    # In practice, you would query biological databases or literature to find the functions of these genes

# 9. Cross-Validation
# Perform cross-validation to validate model performance
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_rf, X_train_resampled, y_train_resampled, cv=skf, scoring='accuracy')

print(f'Cross-Validation Accuracy Scores for Random Forest: {cv_scores}')
print(f'Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}')

# 10. Saving the Best Model (Optional)
import joblib

# Save the trained model to a file
joblib.dump(best_rf, 'best_random_forest_model.pkl')

# 11. Loading the Model and Making Predictions (Optional)
# Load the model from the file
loaded_model = joblib.load('best_random_forest_model.pkl')

# Use the loaded model to make predictions
sample_prediction = loaded_model.predict(X_test[:5])
print('Sample Predictions:')
print(sample_prediction)
print('Actual Classes:')
print(y_test[:5].values)
