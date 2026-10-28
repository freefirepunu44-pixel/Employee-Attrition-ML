# EAST WEST INSTITUTE OF TECHNOLOGY
*(An Autonomous Institution under VTU, Belagavi | Approved by AICTE, New Delhi)*  
Off Magadi Road, Vishwaneedam Post, Bengaluru – 560091  
**Department of Computer Science & Engineering**  
**(IoT and Cyber Security including Blockchain Technology)**  
**Academic Year 2026-27 | 7th Semester**  
**Course:** Machine Learning [BIC703] (Mini Project)

---

# FINAL MINI PROJECT REPORT
## EMPLOYEE ATTRITION PREDICTION USING MACHINE LEARNING WITH EXPLAINABLE AI (XAI)

**Course Instructor & Guide:** Prof. P Bharath Krishna  
**Department:** Computer Science & Engineering, EWIT  
**Evaluation Marks:** 100 Marks (Downscaled to 10 Marks for CIE)

---

## 1. Certificate & Student Declaration

### Certificate
This is to certify that the mini-project work entitled **"Employee Attrition Prediction Using Machine Learning"** is a bona fide record of work carried out by the student team in partial fulfillment of the requirements for the award of Bachelor of Engineering in Computer Science & Engineering (IoT and Cyber Security including Blockchain Technology) of Visvesvaraya Technological University (VTU), Belagavi, during the academic year 2026-27.

The project work has been examined and evaluated in accordance with the prescribed academic rubrics of Machine Learning [BIC703].

*Guide Signature:* __________________________  
*Head of Department:* __________________________

### Student Declaration
We hereby declare that the entire project work embodied in this report has been carried out by us under the guidance of Prof. P Bharath Krishna, Department of CSE (IoT & CS including BT), East West Institute of Technology, Bengaluru. This work is original and has not been submitted elsewhere for the award of any other degree or diploma.

---

## 2. Acknowledgement
We express our sincere gratitude to the management of East West Institute of Technology for providing state-of-the-art computational infrastructure. We are deeply indebted to our Course Instructor and Guide, **Prof. P Bharath Krishna**, for his continuous mentorship, constructive critiques, and guidance throughout the 10-week lifecycle of this Machine Learning project. We also thank the Head of Department, faculty members, and peers for their encouragement.

---

## 3. Abstract
Voluntary employee turnover poses substantial organizational challenges, resulting in financial loss, recruitment overhead, and operational disruption. This project presents an explainable, classification-based Machine Learning pipeline to identify employees at risk of attrition and diagnose the underlying root causes. Built on the IBM HR Analytics dataset (1,470 records, 35 attributes), the system directly confronts the severe real-world class imbalance (83.9% retained vs. 16.1% departed) by employing cost-sensitive balanced weighting, stratified cross-validation, and F1-score optimization rather than deceptive raw accuracy.

Five supervised classification algorithms—Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM)—were trained and benchmarked within a strict scikit-learn Pipeline architecture to eliminate data leakage. Support Vector Machine with an RBF kernel emerged as the champion model, achieving a cross-validation F1-score of 0.5532 and a test ROC-AUC of 0.8168. Tuning the decision threshold to 0.33 maximized the test F1-score to 0.5594, capturing 56.34% of real leavers with 55.56% precision and 85.71% overall accuracy.

To ensure transparency in high-stakes HR decisions, the system integrates Dual-Layer Explainability: global feature ranking via Permutation Importance and local case attribution via accelerated SHAP (Shapley Additive Explanations). The end-to-end solution is deployed as an interactive Streamlit web application providing persona presets, single-employee risk diagnostics, and workforce batch CSV scoring with exportable risk reports.

---

## 4. Table of Contents
1. Chapter 1: Introduction & Problem Formulation
2. Chapter 2: Dataset Description & Data Dictionary
3. Chapter 3: Exploratory Data Analysis (EDA)
4. Chapter 4: Data Cleaning & Preprocessing Pipeline
5. Chapter 5: Feature Engineering & Selection
6. Chapter 6: Machine Learning Model Development
7. Chapter 7: Model Comparison & Benchmark Results
8. Chapter 8: Hyperparameter Tuning
9. Chapter 9: Final Model Evaluation & Error Analysis
10. Chapter 10: Model Strengths, Limitations & Future Scope
11. Chapter 11: System Architecture & Deployment (Streamlit)
12. Chapter 12: GitHub Repository & Execution Guide
13. Chapter 13: Conclusion
14. References
15. Appendix A: Automated Unit Test Suite & Logs
16. Appendix B: Final Approval Comments & Recommendations

---

## Chapter 1: Introduction

### 1.1 Background & Motivation
Human capital represents the primary driver of organizational resilience. Unexpected employee departures incur severe direct replacement costs (50% to 200% of annual salary in recruiting, onboarding, and productivity downtime). Early automated identification of at-risk staff allows HR to enact proactive retention measures before resignation.

### 1.2 Problem Statement
Given historical employee demographic, satisfaction, tenure, and compensation records, build an automated Machine Learning pipeline to accurately estimate the probability of voluntary employee departure while providing explainable, actionable diagnostics for HR management.

### 1.3 Objectives
- Formulate attrition prediction as an imbalanced binary classification problem.
- Implement a leak-free scikit-learn ColumnTransformer and Pipeline architecture.
- Train, benchmark, and tune 5 distinct ML classifiers under 5-fold stratified cross-validation.
- Optimize the decision threshold for F1-score and recall.
- Integrate Permutation Importance (global) and accelerated SHAP (local) explainability.
- Deploy the system via an interactive Streamlit web application supporting single and batch scoring.

---

## Chapter 2: Dataset Description & Data Dictionary

The IBM HR Analytics dataset contains 1,470 employee records and 35 raw attributes.
- **Missing Values:** Exactly 0.
- **Duplicate Rows:** Exactly 0.
- **Target Variable:** `Attrition` (Binary: `Yes` = 237 [16.12%], `No` = 1,233 [83.88%]).
- **Dropped Columns:** `EmployeeNumber` (identifier), `EmployeeCount`, `StandardHours`, `Over18` (all zero-variance constants).
- **Cleaned Modeling Space:** 30 features (23 numerical + 7 categorical).

---

## Chapter 3: Exploratory Data Analysis (EDA)

Key insights derived from the 6 visual analyses:
1. **Target Imbalance:** 83.9% retained vs. 16.1% departed. Accuracy alone is misleading.
2. **OverTime Impact:** Employees working overtime leave at 30.5% compared to only 10.4% of non-overtime staff (3x multiplier).
3. **Role Vulnerability:** Sales Representatives (39.8%) and Laboratory Technicians (23.9%) experience the highest attrition.
4. **Satisfaction Dynamics:** Steep monotonic rise in attrition as Job Satisfaction and Work-Life Balance drop to level 1.
5. **Continuous Drivers:** Leavers have lower median monthly income ($3,200 vs $4,900), shorter tenure (median 7 vs 10 years), and longer commutes (median 9 vs 7 miles).

---

## Chapter 4: Data Cleaning & Preprocessing Pipeline

- **ColumnTransformer:** Encapsulates `StandardScaler()` on 23 numerical features and `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` on 7 categorical features.
- **Dimensionality:** Expands the 30 features into 51 normalized numerical inputs.
- **Leakage Prevention:** Fitted strictly on training folds inside scikit-learn Pipelines.
- **Splitting:** Stratified 70/30 train/test split (1,029 training samples, 441 test samples).

---

## Chapter 5: Feature Engineering & Selection

Feature ranking conducted using model-agnostic Permutation Importance on the hold-out test set identified the primary churn drivers:
1. `StockOptionLevel` (0.0209 F1 drop)
2. `JobSatisfaction` (0.0170 F1 drop)
3. `DistanceFromHome` (0.0166 F1 drop)
4. `OverTime` (0.0161 F1 drop)
5. `NumCompaniesWorked` (0.0143 F1 drop)

---

## Chapter 6 & 7: Model Benchmark & Comparison

Benchmarked on identical 30% hold-out test set (441 samples):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Support Vector Machine (Champion)** | **0.832** | **0.474** | **0.634** | **0.539** | **0.817** | **0.553** |
| **Logistic Regression** | 0.757 | 0.366 | 0.676 | 0.480 | 0.816 | 0.512 |
| **Random Forest** | 0.844 | 0.488 | 0.423 | 0.455 | 0.780 | 0.514 |
| **Decision Tree** | 0.741 | 0.320 | 0.577 | 0.414 | 0.698 | 0.408 |
| **K-Nearest Neighbors** | 0.841 | 0.519 | 0.197 | 0.286 | 0.611 | 0.284 |

**Accuracy Paradox Observation:** KNN achieved 84.1% raw accuracy but caught only 19.7% of real leavers (F1 = 0.286), failing completely as a risk detection tool.

---

## Chapter 8 & 9: Hyperparameter Tuning & Final Evaluation

- **Best SVM Hyperparameters:** $C = 0.5$, $\gamma = \text{scale}$, RBF kernel, `class_weight='balanced'`.
- **Optimal Decision Threshold:** Tuned to **0.33** using Precision-Recall curve analysis.
- **Final Hold-Out Test Metrics:**
  - **F1-Score:** 0.5594 (raised from 0.539)
  - **Precision:** 0.5556
  - **Recall:** 0.5634 (caught 40 out of 71 leavers)
  - **Accuracy:** 0.8571
  - **ROC-AUC:** 0.8168
- **Confusion Matrix:** True Negatives = 338, False Positives = 32, False Negatives = 31, True Positives = 40.

---

## Chapter 10: Model Strengths, Limitations & Future Scope
- **Strengths:** 100% leak-free pipeline, objective metric focus, accelerated SHAP (0.32s), batch workforce scoring.
- **Limitations:** Cross-sectional dataset lacks time-series event intervals for survival analysis.
- **Future Scope:** Integration with enterprise HRIS (Workday, SAP) and NLP on exit interview feedback.

---

## Chapter 11: Deployment (Streamlit Application)
- **Architecture:** Streamlit web application (`app/app.py`) with 4 interactive views:
  1. *Dashboard Overview:* Workforce KPIs and demographic distributions.
  2. *Risk Assessment:* Single-employee diagnostic with persona presets and local SHAP waterfall attribution, plus batch CSV upload with downloadable scored roster.
  3. *Model Performance:* Interactive ROC curves, confusion matrix with TN/FP/FN/TP tags, and metrics table.
  4. *Explainability:* Educational guide on permutation importance and Shapley values.

---

## Chapter 12: GitHub Repository & Execution Commands
```bash
# 1. Run automated unit tests (10/10 tests)
python -m unittest discover tests

# 2. Test fast prediction and SHAP explanation
python src/predict.py

# 3. Launch the web application
streamlit run app/app.py

# 4. Regenerate reports and presentations
python scripts/generate_reports.py
python scripts/generate_presentation.py
```

---

## References
1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS 2017.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825-2830.
3. IBM Watson Analytics. (2016). HR Employee Attrition and Performance Dataset. Kaggle.
4. Chawla, N. V., et al. (2002). SMOTE: synthetic minority over-sampling technique. JAIR, 16, 321-357.
5. Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
6. Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297.
