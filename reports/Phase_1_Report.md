# EAST WEST INSTITUTE OF TECHNOLOGY
*(An Autonomous Institution under VTU, Belagavi | Approved by AICTE, New Delhi)*  
Off Magadi Road, Vishwaneedam Post, Bengaluru – 560091  
**Department of Computer Science & Engineering**  
**(IoT and Cyber Security including Blockchain Technology)**  
**Academic Year 2026-27 | 7th Semester**  
**Course:** Machine Learning [BIC703] (Mini Project)

---

# PHASE 1 EVALUATION REPORT (40 MARKS)
## EMPLOYEE ATTRITION PREDICTION USING MACHINE LEARNING

**Course Instructor & Guide:** Prof. P Bharath Krishna  
**Department:** Computer Science & Engineering, EWIT  
**Phase 1 Timeline:** Weeks 1 to 5 (17th Aug 2026 – 14th Sep 2026)

---

## 1. Project Title
**Employee Attrition Prediction Using Machine Learning With Explainable AI (XAI)**

## 2. Team Details
- **Student Project Team:** USN 1EW23CS001, 1EW23CS002
- **Program & Semester:** B.E. 7th Semester, Department of CSE (IoT and Cyber Security including Blockchain Technology)
- **Institution:** East West Institute of Technology, Bengaluru
- **Course Instructor & Guide:** Prof. P Bharath Krishna

## 3. Problem Statement & Objectives
- **Problem Statement:** Voluntary employee turnover creates significant institutional disruptions, loss of domain knowledge, and recruitment overhead. This project builds an automated machine learning system to estimate the probability of attrition from demographic, satisfaction, tenure, and compensation factors.
- **Objectives:**
  1. Formulate attrition prediction as an imbalanced binary classification problem.
  2. Perform comprehensive exploratory data analysis with minimum 5 visualizations.
  3. Construct a leak-free scikit-learn ColumnTransformer and Pipeline architecture.
  4. Develop and evaluate 5 initial machine learning models using stratified cross-validation.

## 4. Problem Formulation
- **Type:** Supervised Binary Classification.
- **Target Variable:** `Attrition` ∈ {0: No [Retained], 1: Yes [Departed]}.
- **Class Imbalance:** 83.88% No vs. 16.12% Yes. Evaluation metrics must prioritize F1-score, precision, recall, and ROC-AUC over raw accuracy.

## 5. Dataset Description & Data Dictionary
- **Source:** IBM HR Analytics Employee Attrition & Performance Benchmark dataset.
- **Size:** 1,470 records and 35 columns.
- **Data Quality:** Zero missing values and zero duplicate records.
- **Cleaned Feature Space:** 30 features (23 numerical + 7 categorical).
- **Dropped Features:** `EmployeeNumber` (identifier), `EmployeeCount`, `StandardHours`, `Over18` (constant features).

## 6. Exploratory Data Analysis (EDA Visualizations)
1. **Target Distribution:** Confirms severe imbalance (1,233 No vs. 237 Yes).
2. **OverTime Disparity:** Employees working overtime leave at ~30.5% vs. ~10.4% for non-overtime staff.
3. **Job Role Churn:** Sales Representatives suffer the highest turnover (39.8%), followed by Lab Techs (23.9%).
4. **Satisfaction Curves:** Direct inverse relationship between Likert ratings (Work-Life Balance, Job Satisfaction) and churn.
5. **Continuous Drivers:** Departing staff have lower median income ($3,200 vs $4,900) and shorter tenure (7 vs 10 years).

## 7. Data Cleaning & Preprocessing Pipeline
- **Categorical Processing:** OneHotEncoder(handle_unknown='ignore') on 7 categorical columns $\rightarrow$ 28 binary features.
- **Numerical Processing:** StandardScaler() on 23 numerical features.
- **Output Dimensions:** 51 total normalized features.
- **Data Leakage Prevention:** ColumnTransformer fitted strictly on training folds inside scikit-learn Pipelines.
- **Split:** Stratified 70/30 train/test partition (1,029 training, 441 test).

## 8. Initial ML Models & Benchmark Results (Hold-Out Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Support Vector Machine (Champion)** | **0.832** | **0.474** | **0.634** | **0.539** | **0.817** | **0.553** |
| **Logistic Regression** | 0.757 | 0.366 | 0.676 | 0.480 | 0.816 | 0.512 |
| **Random Forest** | 0.844 | 0.488 | 0.423 | 0.455 | 0.780 | 0.514 |
| **Decision Tree** | 0.741 | 0.320 | 0.577 | 0.414 | 0.698 | 0.408 |
| **K-Nearest Neighbors** | 0.841 | 0.519 | 0.197 | 0.286 | 0.611 | 0.284 |

## 9. Challenges & Observations
- **Accuracy Paradox:** KNN achieved 84.1% accuracy but only 19.7% recall, failing to catch 80% of leavers.
- **Feature Expansion:** 51 dimensions favored RBF SVM maximum-margin separation over distance-based KNN.

## 10. Phase 2 Plan (Weeks 6 to 10)
- **Week 6:** Deeper experimentation and probability calibration.
- **Week 7:** Comprehensive hyperparameter tuning using 5-fold Stratified GridSearchCV.
- **Week 8:** Decision threshold tuning (Precision-Recall optimization) and error analysis.
- **Week 9:** Web application deployment via Streamlit with accelerated SHAP local explainability.
- **Week 10:** Final thesis soft-binding report, presentation, and viva defense.

## 11. Guide Evaluation Comments
- **Phase 1 Marks Awarded:** ________ / 40 Marks  
- **Guide Signature:** __________________________  
- **Date:** __________________________
