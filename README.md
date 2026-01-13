# 🩺 Project Name: Chronic Kidney Disease (CKD) Analysis & Prediction

## 🔍 Overview
This project aims to analyze and predict **Chronic Kidney Disease (CKD)** status using clinical and laboratory data.
The dataset contains real-world medical measurements with substantial missingness, making it suitable for evaluating
**statistical inference, predictive modeling, and robustness under different missing data mechanisms**.

The project was completed as a **final project for AMS 572 (Stony Brook University)** and emphasizes
both **statistical interpretability** and **model stability**, rather than black-box prediction alone.

---

## 📈 Results and Key Insights

### 🔑 Important Predictors of CKD
- **Blood Glucose (bgr)**  
  Higher random blood glucose levels were strongly associated with increased CKD risk.
- **Hemoglobin (hemo)**  
  Lower hemoglobin levels significantly increased the probability of CKD.
- **Specific Gravity (sg)**  
  Lower urine specific gravity values were associated with kidney dysfunction.

> These variables remained stable predictors even under simulated missing data scenarios.

---

### ⚠️ Statistical Modeling Insight: Separation Issue
- The variable **Albumin (al)** exhibited **quasi-complete separation** with respect to CKD status.
- Including this variable caused instability in logistic regression estimates.
- **Decision**: Albumin was excluded from the final model to preserve inferential validity.

---

### 🧪 Hypothesis Testing Result (H₁)
- **Hypothesis**: Mean blood pressure differs by appetite status.
- **Method**: Welch two-sample t-test
- **Result**: Statistically significant difference observed (p < 0.01).
- **Robustness**: The significance weakened as missingness increased beyond 30% under MCAR.

---

## 📊 Model Performance

- **Final Model**: Logistic Regression (3 predictors)
- **Accuracy**: ~96%
- **ROC AUC**: ~0.99
- **Sensitivity / Specificity**: Both high and well-balanced

The model demonstrated strong discriminative ability and robustness under MCAR and MNAR simulations.

---

## 📋 Table of Contents
1. Data Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Hypothesis Testing  
4. Logistic Regression Modeling  
5. Missing Data Simulation (MCAR / MNAR)  
6. Model Evaluation & ROC Analysis  
7. Conclusion & Future Work  

---

## 🛠️ 1. Data Preprocessing
- Replaced invalid entries with NA
- Converted categorical variables using dummy encoding
- Standardized numeric variables
- Applied **KNN Imputation (k = 5)** for missing values
- Removed variables with modeling instability

---

## 📊 2. Exploratory Data Analysis (EDA)
- Target variable distribution (CKD vs non-CKD)
- Missingness patterns by variable
- Correlation analysis among numerical features
- Visual diagnostics for separation and outliers

---

## 🔧 3. Model Building
We focused on **interpretable statistical modeling** rather than black-box prediction.

- Logistic Regression
- Variance Inflation Factor (VIF) diagnostics
- Stepwise AIC-based variable selection
- Separation diagnostics and correction

---

## 📉 4. Missing Data Robustness Analysis
To assess inferential stability, we simulated missingness under:

- **MCAR** (10%–50%)
- **MNAR** (value-dependent missingness)

Model coefficients and predictive performance remained stable across scenarios,
demonstrating robustness of the final model.

---

## 📈 5. ROC Curve Visualization

<img width="635" height="473" alt="image" src="https://github.com/user-attachments/assets/93c2df2a-b9c0-41d5-852d-49787ec441ed" />


The ROC curve shows excellent discrimination capability of the final logistic model.

---

## 📝 6. Conclusion
This project demonstrates that:
- Thoughtful variable selection is critical in clinical modeling
- Statistical diagnostics (e.g., separation, multicollinearity) matter
- Simple, interpretable models can outperform complex approaches when properly validated
- Robustness to missing data is essential in healthcare analytics

---

## 🔮 Future Work
- External validation on independent CKD datasets
- Extension to survival analysis for CKD progression
- Comparison with penalized regression methods (LASSO / Ridge)

---

##  Team Members
- **Alan Rodriguez**  
  Stony Brook University
  Applied Mathematics & Statistics (AMS)

  - **Xiaoyan Lin**  
  Stony Brook University
  Applied Mathematics & Statistics (AMS)

- **Yeonbi Han**  
  Stony Brook University  
  Applied Mathematics & Statistics (AMS)


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
└── README.md
