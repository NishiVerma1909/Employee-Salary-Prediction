# Employee-Salary-Prediction
# Income Classification using XGBoost

This project predicts whether an individual's income exceeds $50K/year using the UCI Adult Income dataset. It applies data preprocessing and uses the XGBoost algorithm for binary classification.

---

## Dataset

- **Source**: [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **Target Variable**: `income` (`<=50K` or `>50K`)
- **Features**:
  - `age`, `workclass`, `education`, `marital.status`, `occupation`, `relationship`
  - `race`, `sex`, `capital.gain`, `capital.loss`, `hours.per.week`, `native.country`

---

## Objective

Build a machine learning pipeline to:
- Preprocess the dataset (handle missing values, encode categorical variables, scale numeric features)
- Train a classifier using XGBoost
- Evaluate the model with various metrics
- Visualize important results (confusion matrix, ROC curve, feature importances)

---

## Technologies & Libraries

- Python 3.x
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn

---

## Workflow

1. **Data Cleaning**
   - Replace `'?'` with `NaN`
   - Drop missing rows

2. **Preprocessing**
   - OneHotEncoding for categorical columns
   - StandardScaling for numeric columns

3. **Modeling**
   - Use `XGBClassifier` in a scikit-learn pipeline
   - Perform train-test split using `stratify=y`

4. **Evaluation**
   - Accuracy score
   - Classification report
   - Confusion matrix
   - ROC-AUC score

5. **Visualization**
   - Confusion matrix heatmap
   - ROC curve
   - Feature importance plot

---

## Example Results

| Metric              | Value (Approx) |
|---------------------|----------------|
| Accuracy            | 0.85           |
| ROC-AUC Score       | 0.90+          |
| Precision/Recall    | Balanced       |

> Note: Results may vary based on preprocessing and tuning.

---

## Visual Output

- Confusion Matrix  
- ROC Curve  
- XGBoost Feature Importances

---


