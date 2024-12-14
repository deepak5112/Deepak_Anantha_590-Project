# **Gene Expression Cancer Classification Using Machine Learning**

## **Project Overview**

This project focuses on classifying cancer subtypes based on gene expression data using various machine learning models. The main goal is to preprocess the data, select relevant features, and apply classification models to predict the correct cancer subtype. Models used include Logistic Regression, Random Forest, SVM, XGBoost, ANN, and Naïve Bayes. The project also addresses class imbalance and evaluates models based on accuracy, precision, recall, and F1-score.

---

## **Requirements**

1. **Python version**: Python 3.7 or higher
2. **Libraries**:
   - `pandas`
   - `numpy`
   - `matplotlib`
   - `seaborn`
   - `scikit-learn`
   - `imblearn`
   - `xgboost`
   - `joblib`

To install the required libraries, run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imblearn xgboost joblib
```

---

## **Project Structure**

- `data.csv`: Gene expression data.
- `labels.csv`: Corresponding cancer subtype labels.
- `mlr v1.py`: The main Python script containing the full pipeline for data processing, feature selection, model training, and evaluation.

---

## **Steps and Features**

### **1. Data Loading and Preparation**

- **Load Data**: Gene expression data and cancer subtype labels are merged for further processing.
- **Class Distribution**: Visualized using a pie chart to show class imbalance.

### **2. Exploratory Data Analysis (EDA)**

- **Missing Values**: Checked for missing values and statistical summary of the data.
- **Gene Expression Distribution**: Visualized using violin plots for a sample of genes.

### **3. Data Preprocessing**

- **Constant Feature Removal**: Features with zero variance are removed.
- **Log Transformation**: Applied to reduce skewness in the data.
- **Standardization**: Features are scaled using `StandardScaler`.

### **4. Feature Selection**

- **Mutual Information**: Top features are selected based on their mutual information scores.
- **Dimensionality Reduction**: PCA and t-SNE are used for visualization of gene expression data.

### **5. Addressing Class Imbalance**

- **SMOTE**: Synthetic Minority Over-sampling Technique is applied to balance class distribution in the training data.

### **6. Model Building and Hyperparameter Tuning**

- **Logistic Regression**, **Random Forest**, **SVM**, **XGBoost**, **ANN**, and **Naïve Bayes** models are implemented.
- **GridSearchCV**: Used for hyperparameter tuning to optimize each model's performance.

### **7. Model Evaluation and Comparison**

- **Metrics**: Accuracy, Precision, Recall, and F1 Score are calculated for each model.
- **Confusion Matrix**: Visualized for Logistic Regression.

### **8. Biological Interpretation**

- **Top Features**: The most important features (genes) are identified, and their biological relevance can be further explored.

### **9. Cross-Validation**

- Cross-validation is applied to assess the generalization performance of the best-performing model.

### **10. Model Saving and Loading (Optional)**

- The trained Random Forest model is saved using `joblib` for future use.

---

## **How to Run the Code**

1. Clone this repository or download the `mlr v1.py` script along with the `data.csv` and `labels.csv` files.
2. Run the script using the command:
   ```bash
   python mlr v1.py
   ```

---

## **Key Takeaways**

- **Best Models**: Random Forest and XGBoost consistently performed well across different metrics.
- **Class Imbalance**: Handled effectively using SMOTE, leading to better classification performance.
- **Feature Selection**: Mutual information and PCA helped reduce the dimensionality of the dataset while retaining important features.
- **Biological Insights**: The project highlights genes that are important for cancer subtype classification, which can be further explored for biological significance.

---

## **Further Improvements**

- **Hyperparameter Tuning**: Further fine-tuning could improve model performance.
- **Data Augmentation**: Additional preprocessing and augmentation techniques can be explored to enhance model robustness.
- **Biological Analysis**: Investigate the top genes identified by Logistic Regression to uncover biological relevance.

---

This README provides an overview of the project and instructions for running the `mlr v1.py` script. If you have any questions, feel free to reach out.