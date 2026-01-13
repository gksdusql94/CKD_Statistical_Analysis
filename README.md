# 🩺 Project Name: Chronic Kidney Disease (CKD) Analysis & Prediction

## 🔍 Overview
This project analyzes and predicts **Chronic Kidney Disease (CKD)** using clinical and laboratory measurements.
The dataset reflects real-world medical data with substantial missingness, making it suitable for studying:

- Statistical inference under missing data
- Model robustness (MCAR / MNAR)
- Interpretability-focused clinical modeling

This project was conducted by **three students at Stony Brook University**
with a shared interest in **data analysis and healthcare analytics**.
The emphasis is placed on **statistical interpretability and robustness** rather than black-box prediction.

---

## 📊 Key Visual Summary

### 1️⃣ CKD Status Distribution
<img width="657" height="468" alt="image" src="https://github.com/user-attachments/assets/1821c6d1-9732-4c2e-ba0c-ac919883e2ab" />


- The dataset shows a clear separation between CKD and non-CKD groups.
- Class imbalance was handled during model evaluation.

---

### 2️⃣ Correlation Heatmap of Numerical Features
<img width="521" height="488" alt="image" src="https://github.com/user-attachments/assets/b272900e-c916-4708-9935-ccf4ced1aece" />


- Strong correlations observed among clinical lab variables.
- Used as a diagnostic tool for multicollinearity and variable selection.

---

### 3️⃣ ROC Curve – Final Logistic Regression Model
<img width="658" height="468" alt="image" src="https://github.com/user-attachments/assets/e8da426e-706f-4f0b-abf2-9c5d9ebfeae0" />


- Excellent discrimination between CKD and non-CKD patients.
- ROC AUC ≈ **0.99**, indicating strong predictive performance.

---

## 📈 Results and Key Insights

### 🔑 Important Predictors
- **Blood Glucose (bgr)**: Higher values → higher CKD risk
- **Hemoglobin (hemo)**: Lower values → higher CKD probability
- **Specific Gravity (sg)**: Lower urine concentration → kidney dysfunction

These predictors remained stable across missing data simulations.

---

### ⚠️ Statistical Issue: Quasi-Complete Separation
- **Albumin (al)** showed quasi-complete separation
- Caused unstable logistic regression estimates
- Excluded from final model to preserve inferential validity

---

## 🧪 Hypothesis Testing
- **H₁**: Mean blood pressure differs by appetite status  
- **Method**: Welch two-sample t-test  
- **Result**: Significant (p < 0.01)  
- **Robustness**: Significance weakened beyond 30% MCAR missingness

---

## 📉 Model Performance
- **Model**: Logistic Regression (interpretable, low-dimensional)
- **Accuracy**: ~96%
- **ROC AUC**: ~0.99
- **Sensitivity / Specificity**: Well-balanced

---

## 📋 Workflow
1. Data Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Hypothesis Testing  
4. Logistic Regression Modeling  
5. Missing Data Simulation (MCAR / MNAR)  
6. Model Evaluation (ROC)  

---

## 🛠️ Data Processing Highlights
- Invalid entries replaced with NA
- Dummy encoding for categorical variables
- Standardization of numeric variables
- KNN Imputation (k = 5)
- VIF and separation diagnostics applied

---

## 🔮 Future Work
- External dataset validation
- Penalized regression comparison (LASSO / Ridge)
- Survival analysis for CKD progression

---

## 👥 Team Members
- **Alan Rodriguez**  
  Stony Brook University – Applied Mathematics & Statistics

- **Xiaoyan Lin**  
  Stony Brook University – Applied Mathematics & Statistics

- **Yeonbi Han**  
  Stony Brook University – Applied Mathematics & Statistics

---

## 📂 Repository Structure
```text
CKD_Chronic/
├── Final_CKD.R
├── CKD_Python_Final/
├── chronic_kidney_disease.csv
├── chronic_kidney_disease_features.csv
├── chronic_kidney_disease_targets.csv
├── ckd_analysis_results.html
├── Report_CKD.pdf
├── figures/
│   ├── ckd_target_distribution.png
│   ├── ckd_correlation_heatmap.png
│   └── ckd_roc_curve.png
└── README.md
