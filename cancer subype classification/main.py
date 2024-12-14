# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# For handling imbalanced data
from imblearn.over_sampling import SMOTE

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading and Preparation
# Option 1: Load data with the first column as index
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
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# 4. Feature Selection
# 4.1 Mutual Information
# Calculate mutual information
mi = mutual_info_classif(X_scaled, y, random_state=42)

# Create a series with feature names and their MI scores
mi_scores = pd.Series(mi, index=features)
mi_scores = mi_scores.sort_values(ascending=False)

# Select top 1000 features based on MI scores
top_features = mi_scores.index[:1000]

# Update X with top features
X_selected = X_scaled[:, mi_scores.index.get_indexer(top_features)]
print(f'Features selected after mutual information: {X_selected.shape[1]}')

# 4.2 Principal Component Analysis (PCA)
# Apply PCA for visualization purposes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_selected)

# Plot PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Gene Expression Data')
plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# 5. Addressing Class Imbalance
# Apply SMOTE to balance the dataset
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_selected, y)

# Display new class counts
resampled_counts = pd.Series(y_resampled).value_counts()
print('Resampled Class Counts:')
print(resampled_counts)

# 6. Model Building
# 6.1 Splitting the Data
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

print(f'Training set size: {X_train.shape[0]}')
print(f'Testing set size: {X_test.shape[0]}')

# 6.2 Support Vector Machine (SVM)
# Initialize SVM model
svm_model = SVC(kernel='linear', probability=True, random_state=42)

# Train the model
svm_model.fit(X_train, y_train)

# Predictions
y_pred_svm = svm_model.predict(X_test)

# 6.3 Random Forest Classifier
# Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)


# 6.4 Gradient Boosting Classifier (XGBoost)
# Initialize LabelEncoder
le = LabelEncoder()

# Fit LabelEncoder on the training labels
le.fit(y_train)

# Transform labels
y_train_encoded = le.transform(y_train)
# No need to transform y_test unless required elsewhere

# Initialize XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Train the model with encoded labels
xgb_model.fit(X_train, y_train_encoded)

# Predictions (numeric labels)
y_pred_xgb_encoded = xgb_model.predict(X_test)

# Inverse transform predictions to get original string labels
y_pred_xgb = le.inverse_transform(y_pred_xgb_encoded)

# 6.5 Artificial Neural Network (ANN)
# Initialize ANN model
ann_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

# Train the model
ann_model.fit(X_train, y_train)

# Predictions
y_pred_ann = ann_model.predict(X_test)

# 6.6 Naïve Bayes Classifier
# Initialize Naïve Bayes model
nb_model = GaussianNB()

# Train the model
nb_model.fit(X_train, y_train)

# Predictions
y_pred_nb = nb_model.predict(X_test)

# 7. Model Evaluation and Comparison
# 7.1 Evaluation Metrics
# Function to calculate metrics
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

# Evaluate SVM
acc_svm, prec_svm, rec_svm, f1_svm = calculate_metrics(y_test, y_pred_svm, 'SVM')

# Evaluate Random Forest
acc_rf, prec_rf, rec_rf, f1_rf = calculate_metrics(y_test, y_pred_rf, 'Random Forest')

# Evaluate XGBoost
acc_xgb, prec_xgb, rec_xgb, f1_xgb = calculate_metrics(y_test, y_pred_xgb, 'XGBoost')

# Evaluate ANN
acc_ann, prec_ann, rec_ann, f1_ann = calculate_metrics(y_test, y_pred_ann, 'ANN')

# Evaluate Naïve Bayes
acc_nb, prec_nb, rec_nb, f1_nb = calculate_metrics(y_test, y_pred_nb, 'Naïve Bayes')

# 7.3 Model Comparison
# Create a DataFrame for comparison
models = ['SVM', 'Random Forest', 'XGBoost', 'ANN', 'Naïve Bayes']
accuracy = [acc_svm, acc_rf, acc_xgb, acc_ann, acc_nb]
precision = [prec_svm, prec_rf, prec_xgb, prec_ann, prec_nb]
recall = [rec_svm, rec_rf, rec_xgb, rec_ann, rec_nb]
f1 = [f1_svm, f1_rf, f1_xgb, f1_ann, f1_nb]

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
# Set the model name as index
comparison_df.set_index('Model', inplace=True)

# Plot the comparison
comparison_df.plot(kind='bar', figsize=(10, 7))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 7.5 Confusion Matrix for the Best Model (Random Forest as an example)
# Confusion Matrix for Random Forest
from sklearn.metrics import ConfusionMatrixDisplay

plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, display_labels=rf_model.classes_, cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7.6 Feature Importance from Random Forest
# Get feature importances
importances = rf_model.feature_importances_

# Create a DataFrame
feature_importances = pd.DataFrame({'Feature': top_features, 'Importance': importances})

# Sort by importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot top 20 features
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(20))
plt.title('Top 20 Feature Importances - Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
