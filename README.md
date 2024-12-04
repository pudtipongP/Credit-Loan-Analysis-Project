# Credit Loan Analysis Project

## 0. Motivation
In the modern financial landscape, accurately predicting loan status is crucial for both financial institutions and loan applicants. This project aims to:
- Develop a robust machine learning model to assess loan eligibility
- Understand key factors influencing loan approval
- Provide insights into credit risk assessment
- Demonstrate the power of data-driven decision-making in lending

## 1. Dataset Overview
- **Source**: Kaggle Dataset by Zaur Begiev
- **Dataset URL**: [Credit Loan Dataset](https://www.kaggle.com/datasets/zaurbegiev/my-dataset)
- **Purpose**: Loan Status Prediction
- **Context**: The dataset contains comprehensive information about loan applicants, capturing various financial and personal attributes that may influence loan approval.

### Dataset Attributes
1. **Loan Identification**
   - Loan ID
   - Customer ID

2. **Loan Characteristics**
   - Loan Status (Target Variable)
   - Current Loan Amount
   - Term (Short/Long)

3. **Financial Indicators**
   - Credit Score
   - Annual Income
   - Monthly Debt
   - Years of Credit History
   - Current Credit Balance
   - Maximum Open Credit

4. **Personal Information**
   - Years in Current Job
   - Home Ownership
   - Purpose of Loan
   - Number of Open Accounts
   - Number of Credit Problems
   - Bankruptcies
   - Tax Liens

## 2. Data Preparation

### 2.1 Initial Data Exploration
- **Dataset**: Credit loan dataset
- **Initial Size**: 100,514 entries, 19 columns
- **Data Types**: 12 float64 columns, 7 object columns

### 2.2 Data Cleaning Steps
1. **Duplicate Removal**
   - Total duplicates: 10,728
   - Reduced dataset size from 100,514 to 89,786 entries

2. **Handling Missing Values**
   - Dropped columns with >50% missing data:
     * "Months since last delinquent"
   - Removed rows with missing values in:
     * Maximum Open Credit
     * Tax Liens
     * Bankruptcies
     * Years in current job
   - Final dataset size: 85,791 entries

### 2.3 Data Transformation
1. **Credit Score Adjustment**
   - Identified anomalies in Credit Score (values > 850)
   - Normalized scores by dividing values > 850 by 10
   - Imputed missing Credit Scores using term-based average

2. **Annual Income Imputation**
   - Used KNN Imputation for missing Annual Income values
   - Maintained original data distribution

3. **Feature Engineering**
   - Grouped "Years in current job" into categories:
     * '0-1 year'
     * '2-3 years'
     * '4-6 years'
     * '7-9 years'
     * '>10 years'

### 2.4 Encoding
- Used Label Encoding for categorical columns:
  * Loan Status
  * Term
  * Years in current job
  * Purpose
  * Home Ownership

## 3. Feature Selection
### Correlation Analysis
Top correlated features with Loan Status:
1. Current Loan Amount (0.220545)
2. Credit Score (0.163087)
3. Term (0.143948)
4. Annual Income (0.043525)

## 4. Model Development

### Methodology
- Tested multiple models:
  * Logistic Regression
  * Decision Tree
  * Random Forest

- Imbalance Handling Methods:
  * No Handling
  * Random Oversampling
  * SMOTE
  * Random Undersampling
  * SMOTEENN

### Key Techniques
- RobustScaler for feature scaling
- Class weight balancing
- Various resampling techniques

## 5. Model Evaluation Metrics
- Precision
- Recall
- F1-score
- Accuracy
- ROC AUC Score

## 6. Key Insights
- Model performance varies significantly with different imbalance handling techniques
- Class weight balancing helps improve model performance
- Different algorithms show varying effectiveness

## 7. Recommendations
1. Consider ensemble methods
2. Experiment with feature engineering
3. Explore more advanced imputation techniques
4. Validate model performance on unseen data

## 8. Limitations
- Limited feature set
- Potential data leakage risks
- Simplified imputation methods

## 9. Future Work
- Implement more advanced feature selection
- Try more complex models (XGBoost, Neural Networks)
- Collect more diverse training data
