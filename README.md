# Employee Attrition Prediction Using Machine Learning

An explainable, classification-based employee risk analysis system built as an academic ML
mini-project. The system trains and compares five classifiers, selects the best one by F1 score
(not raw accuracy), and exposes the result through an interactive Streamlit dashboard with
SHAP-based explanations for every prediction.

## What it does

- Trains **Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors and SVM**
  on the IBM HR Analytics Employee Attrition dataset (1,470 records, 35 columns).
- Handles the natural **class imbalance** (~84% / 16%) with `class_weight="balanced"`,
  stratified splitting, and evaluation focused on **F1, recall and ROC-AUC** rather than accuracy.
- **Tunes hyperparameters** with `GridSearchCV` / `RandomizedSearchCV` under 5-fold stratified
  cross-validation.
- Picks the **best model by hold-out F1** and saves a scikit-learn `Pipeline` (scaling + one-hot
  encoding + classifier) so raw categorical inputs predict directly with **no data leakage**.
- Explains predictions with **permutation importance** (global) and **accelerated SHAP** (local).
- Provides an **interactive Streamlit web dashboard** supporting both individual employee assessment (with persona presets) and batch workforce CSV scoring with report downloads.

## Results (hold-out test set, 30%)

| Model                  | Accuracy | Precision | Recall | F1    | ROC-AUC | CV F1 |
| ---------------------- | -------- | --------- | ------ | ----- | ------- | ----- |
| Support Vector Machine | 0.83     | 0.47      | 0.63   | 0.539 | 0.817   | 0.553 |
| Logistic Regression    | 0.76     | 0.37      | 0.68   | 0.480 | 0.816   | 0.512 |
| Random Forest          | 0.84     | 0.49      | 0.42   | 0.455 | 0.780   | 0.514 |
| Decision Tree          | 0.74     | 0.32      | 0.58   | 0.414 | 0.698   | 0.408 |
| K-Nearest Neighbors    | 0.84     | 0.52      | 0.20   | 0.286 | 0.611   | 0.284 |

Notice that **KNN has the highest raw accuracy (84%) but the lowest F1 (0.286)** - it catches only 20% of leavers. This is the **Accuracy Paradox** in imbalanced classification.

**Final Champion Model: Support Vector Machine** (RBF kernel, C=0.5, gamma=scale). Applying an optimal decision
threshold of **0.33** (tuned to maximize F1) raises test scores to **F1=0.559, precision=0.56,
recall=0.56, accuracy=0.86** while keeping ROC-AUC at 0.817. Full per-model metrics are in `outputs/model_results.json`.

## Project Structure

```
Employee-Attrition-ML/
├── data/
│   ├── employee_attrition.csv            # IBM HR Analytics dataset (1,470 records)
│   └── DATA_DICTIONARY.md                # Comprehensive 35-variable data dictionary
├── notebooks/
│   └── exploratory_analysis.ipynb        # Fully executed EDA notebook with embedded plots
├── src/
│   ├── preprocessing.py                  # Data loading, column definitions, ColumnTransformer
│   ├── train.py                          # 5-model training, hyperparameter tuning & artifacts
│   ├── evaluate.py                       # Evaluation metrics, confusion matrix & ROC curves
│   └── predict.py                        # Probability inference, batch scoring & fast SHAP
├── models/
│   └── best_model.pkl                    # Serialized pipeline bundle (model + preprocessor + explainer)
├── outputs/
│   ├── model_results.json                # Benchmark metrics, ROC coordinates & confusion matrices
│   └── plots/                            # confusion_matrix, roc_curves, model_comparison, feature_importance
│       └── eda/                          # 6 high-resolution publication-grade EDA figures
├── app/
│   └── app.py                            # Streamlit dashboard (Overview, Predict, Models, Explainability)
├── reports/
│   ├── Final_Project_Report.docx         # 22-chapter thesis formatted per EWIT BIC703 rubrics
│   ├── Final_Project_Report.md           # Markdown version of complete project report
│   ├── Phase_1_Report.docx               # Phase 1 Evaluation report (Weeks 1-5, 40 marks)
│   ├── Phase_1_Report.md                 # Markdown version of Phase 1 report
│   └── Phase_1_Presentation.pptx         # Presentation deck for Phase 1 viva
├── scripts/
│   ├── generate_reports.py               # Generates DOCX reports with embedded figures and tables
│   ├── generate_eda_plots.py             # Generates all 6 EDA figures
│   ├── generate_presentation.py          # Generates final and phase 1 presentation decks
│   └── execute_notebook.py               # Headless EDA notebook execution engine
├── tests/
│   └── test_pipeline.py                  # Automated unit test suite (10/10 passing tests)
├── Employee_Attrition_Presentation.pptx   # 25-slide final presentation deck
├── VIVA_PREPARATION.md                   # Comprehensive viva defense & technical interview guide
├── requirements.txt                      # Project dependencies
└── README.md
```

## Setup & Quickstart

```bash
cd Employee-Attrition-ML
pip install -r requirements.txt
```

## Reproduce the Pipeline

```bash
# 1. Run unit test suite (10/10 tests)
python -m unittest discover tests

# 2. View or execute the EDA notebook
jupyter notebook notebooks/exploratory_analysis.ipynb
# or run headless: python scripts/execute_notebook.py

# 3. Train all models, tune, select best, save artifacts + plots
python src/train.py

# 4. Test the prediction + SHAP service from the CLI
python src/predict.py

# 5. Regenerate the 25-slide PowerPoint deck
python scripts/generate_presentation.py

# 6. Launch the Streamlit dashboard
streamlit run app/app.py
```

## Dashboard Features

1. **Overview / Dashboard** — Workforce KPIs, 84/16 class split, overtime impact, departmental breakdown, work-life balance curves, and global permutation feature importance.
2. **Risk Assessment** — 
   - **Single Employee Assessment:** Pre-filled persona presets (High Risk, Low Risk, Average) or manual 30-feature tuning; real-time gauge, risk tier (Low, Medium, High, Critical), verdict, and instant SHAP local explanation chart.
   - **Batch Workforce Assessment (CSV):** Upload any HR employee CSV or load 100 sample records; generates department risk distribution, high-risk employee rankings, and one-click download of the scored CSV.
3. **Model Performance** — Head-to-head metrics table, ROC curves with AUC annotations, interactive confusion matrix (with TN, FP, FN, TP tags), and F1 vs ROC-AUC scatter plot.
4. **Explainability** — Global permutation importance vs local additive SHAP explanation guide.

## Viva & Project Defense Guide

For in-depth explanations of all design decisions, mathematical formulations, and evaluation trade-offs, refer to [VIVA_PREPARATION.md](VIVA_PREPARATION.md).