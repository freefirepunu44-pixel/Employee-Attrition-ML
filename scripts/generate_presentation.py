import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PLOTS = OUT / "plots"
JSON_RESULTS = OUT / "model_results.json"
PREZ_PATH = ROOT / "Employee_Attrition_Presentation.pptx"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
BLUE = RGBColor(0x2F, 0x66, 0x90)
ORANGE = RGBColor(0xF4, 0xA2, 0x59)
DARK = RGBColor(0x2B, 0x2B, 0x2B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0x6B, 0x6B, 0x6B)

prs = Presentation()
prs.slide_width = 12192000
prs.slide_height = 6858000
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def add_bg(s, color=WHITE):
    fill = s.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def textbox(s, x, y, w, h):
    box = s.shapes.add_textbox(x, y, w, h)
    return box.text_frame, box


def title_bar(s, text):
    rect = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, 900000)
    rect.fill.solid()
    rect.fill.fore_color.rgb = NAVY
    rect.line.fill.background()
    tf = rect.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = 2000000
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.LEFT
    rect.text_frame.margin_left = 400000
    return rect


def bullets(s, items, x, y, w, h, size=1400000, color=DARK, spacing=1.0):
    tf, _ = textbox(s, x, y, w, h)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(item, tuple):
            lead, rest = item
            run = p.add_run()
            run.text = lead + " "
            run.font.bold = True
            run.font.size = size
            run.font.color.rgb = NAVY
            run2 = p.add_run()
            run2.text = rest
            run2.font.size = size
            run2.font.color.rgb = color
        else:
            p.text = item
            p.font.size = size
            p.font.color.rgb = color
        p.space_after = int(60000 * spacing)
    return tf


def section_slide(title, subtitle):
    s = slide()
    add_bg(s, NAVY)
    rect = s.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 2400000, prs.slide_width, 2000000
    )
    rect.fill.background()
    rect.line.fill.background()
    tf = rect.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = 3000000
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER
    tf2, _ = textbox(s, 1500000, 4600000, 9200000, 700000)
    p2 = tf2.paragraphs[0]
    p2.text = subtitle
    p2.font.size = 1500000
    p2.font.color.rgb = ORANGE
    p2.alignment = PP_ALIGN.CENTER
    return s


def content_slide(title, image_path=None, image_x=500000, image_w=8200000,
                  caption=None, image_y=1100000):
    s = slide()
    add_bg(s)
    title_bar(s, title)
    if image_path:
        pic = s.shapes.add_picture(str(image_path), image_x, image_y, width=image_w)
        if caption:
            tf, _ = textbox(s, image_x, pic.top + pic.height + 200000, image_w, 400000)
            p = tf.paragraphs[0]
            p.text = caption
            p.font.size = 1100000
            p.font.color.rgb = GRAY
            p.alignment = PP_ALIGN.CENTER
    return s


def add_footer(s, text="Employee Attrition Prediction | ML Mini-Project"):
    tf, _ = textbox(s, 300000, 6450000, prs.slide_width - 600000, 300000)
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = 900000
    p.font.color.rgb = GRAY
    p.alignment = PP_ALIGN.CENTER


# ---------------------------------------------------------------------------
# 1. Title slide
# ---------------------------------------------------------------------------
s = slide()
add_bg(s, NAVY)
rect = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 1000000, prs.slide_width, 2400000)
rect.fill.background()
rect.line.fill.background()
tf = rect.text_frame
tf.word_wrap = True

p_col = tf.paragraphs[0]
p_col.text = "EAST WEST INSTITUTE OF TECHNOLOGY"
p_col.font.size = 2200000
p_col.font.bold = True
p_col.font.color.rgb = ORANGE
p_col.alignment = PP_ALIGN.CENTER

p_dept = tf.add_paragraph()
p_dept.text = "Department of CSE (IoT and Cyber Security including Blockchain Technology)"
p_dept.font.size = 1300000
p_dept.font.color.rgb = WHITE
p_dept.alignment = PP_ALIGN.CENTER

p = tf.add_paragraph()
p.text = "\nEMPLOYEE ATTRITION PREDICTION USING ML"
p.font.size = 3200000
p.font.bold = True
p.font.color.rgb = WHITE
p.alignment = PP_ALIGN.CENTER

p2 = tf.add_paragraph()
p2.text = "An Explainable Classification-Based Employee Risk Analysis System"
p2.font.size = 1500000
p2.font.color.rgb = ORANGE
p2.alignment = PP_ALIGN.CENTER

tf2, _ = textbox(s, 1500000, 4800000, 9200000, 1500000)
p3 = tf2.paragraphs[0]
p3.text = "Machine Learning [BIC703] Mini Project | 7th Semester | Academic Year 2026-27"
p3.font.size = 1300000
p3.font.color.rgb = WHITE
p3.alignment = PP_ALIGN.CENTER

p4 = tf2.add_paragraph()
p4.text = "Course Instructor & Guide: Prof. P Bharath Krishna | Assistant Professor, Dept of CSE"
p4.font.size = 1250000
p4.font.color.rgb = ORANGE
p4.alignment = PP_ALIGN.CENTER

add_footer(s, "East West Institute of Technology | Machine Learning [BIC703]")

# ---------------------------------------------------------------------------
# 2. Agenda
# ---------------------------------------------------------------------------
s = slide()
add_bg(s)
title_bar(s, "Agenda")
bullets(s, [
    ("1.", "Problem Statement & Objectives"),
    ("2.", "Dataset Overview & Exploratory Data Analysis"),
    ("3.", "Machine Learning Pipeline & Preprocessing"),
    ("4.", "Models, Hyperparameter Tuning & Evaluation"),
    ("5.", "Results: Model Comparison"),
    ("6.", "Explainability (Permutation Importance + SHAP)"),
    ("7.", "Interactive Dashboard"),
    ("8.", "Conclusion, Future Work & Viva Q&A"),
], 900000, 1300000, 10000000, 4500000)
add_footer(s)

# ---------------------------------------------------------------------------
# 3. Problem statement
# ---------------------------------------------------------------------------
s = slide()
add_bg(s)
title_bar(s, "Problem Statement")
bullets(s, [
    ("Question:", "Can we reliably predict whether an employee will leave the company (attrition) before it happens?"),
    ("Goal:", "Build a binary classifier that estimates the probability of attrition from HR attributes."),
    ("Why it matters:", "Proactive retention programs reduce hiring and training costs."),
    ("Core challenge:", "Attrition is rare (~16%) - the dataset is imbalanced, so accuracy alone is misleading."),
    ("Deliverable:", "A dashboard that explains WHY a specific employee is at risk, not just a single label."),
], 900000, 1300000, 10000000, 4500000)
add_footer(s)

# ---------------------------------------------------------------------------
# 4. Dataset overview
# ---------------------------------------------------------------------------
s = slide()
add_bg(s)
title_bar(s, "Dataset Overview - IBM HR Analytics")
data = json.loads(JSON_RESULTS.read_text(encoding="utf-8"))
metrics_line = "1470 employee records | 35 attributes | 0 missing values | 0 duplicates"
tf, _ = textbox(s, 900000, 1300000, 10000000, 500000)
p = tf.paragraphs[0]
p.text = metrics_line
p.font.size = 1600000
p.font.color.rgb = BLUE
p.font.bold = True

bullets(s, [
    ("Target:", "Attrition (Yes / No) - binary classification."),
    ("Features (30 after cleaning):", "Age, JobRole, MonthlyIncome, OverTime, JobSatisfaction, YearsAtCompany, DistanceFromHome, WorkLifeBalance, EnvironmentSatisfaction, JobInvolvement, TotalWorkingYears, and more."),
    ("Removed as irrelevant:", "EmployeeNumber (ID), EmployeeCount, Over18, StandardHours (all constant)."),
    ("Class balance:", "No = 1233 (83.9%), Yes = 237 (16.1%)."),
    ("Health:", "No missing values, no duplicate rows - minimal cleaning required."),
], 900000, 2000000, 10000000, 3500000)
add_footer(s)

# ---------------------------------------------------------------------------
# 5. EDA section
# ---------------------------------------------------------------------------
section_slide("Exploratory Data Analysis", "What the data tells us before modelling")

s = slide()
add_bg(s)
title_bar(s, "EDA Findings - Key Drivers of Attrition")
bullets(s, [
    ("1. Class imbalance", "Yes = 16.1% of rows; evaluation must use F1 / ROC-AUC."),
    ("2. Overtime", "Employees working overtime leave at roughly double the rate."),
    ("3. Low satisfaction", "Job, environment and work-life balance satisfaction fall before exits."),
    ("4. Tenure & age", "Younger employees and short-tenure staff churn more."),
    ("5. Distance from home", "Longer commutes correlate with higher attrition."),
    ("6. Salary:", "MonthlyIncome alone has weak correlation - interactions matter (handled by tuned models & permutation importance)."),
], 900000, 1300000, 10000000, 4500000)
add_footer(s)

# ---------------------------------------------------------------------------
# 6. Methodology section
# ---------------------------------------------------------------------------
section_slide("Methodology", "A rigorous, reproducible ML pipeline")

s = slide()
add_bg(s)
title_bar(s, "Machine Learning Pipeline")
steps = [
    "Load dataset",
    "Drop irrelevant columns",
    "Train / test split (70/30, stratified)",
    "Preprocess inside a sklearn Pipeline",
    "Train 5 baseline classifiers (CV)",
    "Hyperparameter tuning (Grid/RandomizedSearchCV)",
    "Evaluate on hold-out test set",
    "Select best model by F1",
    "Explain with permutation importance + SHAP",
]
for i, step in enumerate(steps):
    rect = s.shapes.add_shape(
        MSO_SHAPE.CHEVRON, 900000, 1000000 + i * 550000, 6000000, 420000
    )
    rect.fill.solid()
    rect.fill.fore_color.rgb = BLUE if i % 2 == 0 else NAVY
    rect.line.fill.background()
    tf = rect.text_frame
    tf.word_wrap = True
    tf.margin_left = 80000
    p = tf.paragraphs[0]
    p.text = f"{i + 1}. {step}"
    p.font.size = 1200000
    p.font.color.rgb = WHITE
add_footer(s, "")

s = slide()
add_bg(s)
title_bar(s, "Preprocessing Without Data Leakage")
bullets(s, [
    ("Split first:", "Train/test split happens BEFORE fitting any transformer."),
    ("ColumnTransformer:", "StandardScaler on numeric, OneHotEncoder on 7 categorical columns."),
    ("One pipeline:", "scaling + encoding + classifier wrapped in one sklearn Pipeline."),
    ("No leakage:", "All preprocessing parameters are learned only from the training fold."),
    ("Input API:", "The final model accepts raw categorical strings - perfect for the dashboard."),
    ("Imbalance:", "class_weight='balanced' used in LR, DT, RF, SVM; SMOTE not forced (16% imbalance is mild, balanced weights already raised recall)."),
], 900000, 1300000, 10000000, 4500000)
add_footer(s)

s = slide()
add_bg(s)
title_bar(s, "Models Compared")
bullets(s, [
    ("1. Logistic Regression", "- interpretable baseline."),
    ("2. Decision Tree", "- easy to explain."),
    ("3. Random Forest", "- strong ensemble candidate."),
    ("4. K-Nearest Neighbors", "- distance-based."),
    ("5. Support Vector Machine", "- margin-based, RBF kernel."),
    ("Tuning:", "GridSearchCV (LR, DT, KNN) + RandomizedSearchCV (RF, SVM), 5-fold stratified CV, scoring = F1."),
    ("Selection:", "best model = highest F1 on the untouched test set."),
], 900000, 1300000, 10000000, 4500000)
add_footer(s)

s = slide()
add_bg(s)
title_bar(s, "Evaluation Metrics - Why Not Only Accuracy?")
bullets(s, [
    ("Accuracy", "overall correctness - misleading on imbalanced data."),
    ("Precision", "of predicted leavers, how many truly left."),
    ("Recall", "of actual leavers, how many we caught."),
    ("F1 score", "harmonic mean balancing precision and recall (our selection metric)."),
    ("ROC-AUC", "ranking quality across all decision thresholds."),
    ("Confusion matrix", "full breakdown of correct / incorrect predictions."),
    ("Key example:", "KNN reached the highest accuracy (84%) yet had the worst F1 - it barely caught any leavers."),
], 900000, 1300000, 10000000, 4500000)
add_footer(s)

# ---------------------------------------------------------------------------
# 7. Results
# ---------------------------------------------------------------------------
section_slide("Results", "Actual metrics computed on the hold-out test set")

s = slide()
add_bg(s)
title_bar(s, "Model Comparison (Hold-out Test Set)")
rows = 6
cols = 7
tbl_x, tbl_y, tbl_w, tbl_h = 700000, 1300000, 10800000, 4300000
shape = s.shapes.add_table(rows, cols, tbl_x, tbl_y, tbl_w, tbl_h)
tbl = shape.table
headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV F1"]
order = {  # display order
    "Support Vector Machine": 0,
    "Logistic Regression": 1,
    "Random Forest": 2,
    "Decision Tree": 3,
    "K-Nearest Neighbors": 4,
}
models = sorted(data["all_models"].items(), key=lambda kv: order.get(kv[0], 5))
for j, h in enumerate(headers):
    cell = tbl.cell(0, j)
    cell.text = h
    cell.text_frame.paragraphs[0].font.bold = True
    cell.text_frame.paragraphs[0].font.size = 1200000
    cell.text_frame.paragraphs[0].font.color.rgb = WHITE
    cell.fill.solid()
    cell.fill.fore_color.rgb = NAVY
for i, (name, m) in enumerate(models, start=1):
    vals = [name, m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"], m["cv_f1_mean"]]
    for j, v in enumerate(vals):
        cell = tbl.cell(i, j)
        if j == 0:
            txt = v
        else:
            txt = f"{v:.3f}"
        cell.text = txt
        cell.text_frame.paragraphs[0].font.size = 1100000
        if name == "Support Vector Machine":
            if j == 4:
                cell.text_frame.paragraphs[0].font.bold = True
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xDF, 0xEA, 0xF3)
add_footer(s, "Highlighted row = selected final model (SVM)")

s = content_slide(
    "Model Comparison - F1 Score",
    image_path=PLOTS / "model_comparison.png",
)
add_footer(s)

s = content_slide(
    "ROC Curves (Best Model AUC = 0.817)",
    image_path=PLOTS / "roc_curves.png",
)
add_footer(s)

fm = {
    "Support Vector Machine (final)": {
        "Accuracy": data["final_metrics"]["accuracy"],
        "Precision": data["final_metrics"]["precision"],
        "Recall": data["final_metrics"]["recall"],
        "F1": data["final_metrics"]["f1"],
        "ROC-AUC": data["final_metrics"]["roc_auc"],
    }
}

s = slide()
add_bg(s)
title_bar(s, "Final Model - Support Vector Machine")
bullets(s, [
    ("Configuration:", "RBF kernel, C = 0.5, gamma = scale (best from RandomizedSearchCV)."),
    ("Imbalance handling:", "class_weight='balanced'."),
    ("Decision threshold:", f"{data['final_metrics']['best_threshold']:.2f} - tuned to maximize F1 on the hold-out set."),
    ("Importance source:", "Permutation importance (SVM has no built-in tree importance)."),
], 900000, 1300000, 6000000, 4000000)

shape2 = s.shapes.add_table(2, 5, 7200000, 1400000, 4600000, 1000000)
tbl2 = shape2.table
mt = fm["Support Vector Machine (final)"]
head = list(mt.keys())
for j, h in enumerate(head):
    c = tbl2.cell(0, j)
    c.text = h
    c.text_frame.paragraphs[0].font.bold = True
    c.text_frame.paragraphs[0].font.size = 1000000
    c.fill.solid()
    c.fill.fore_color.rgb = NAVY
    c.text_frame.paragraphs[0].font.color.rgb = WHITE
    c2 = tbl2.cell(1, j)
    c2.text = f"{mt[h]:.3f}"
    c2.text_frame.paragraphs[0].font.size = 1000000
    c2.text_frame.paragraphs[0].font.color.rgb = BLUE
add_footer(s)

s = content_slide(
    "Confusion Matrix - Final Model",
    image_path=PLOTS / "confusion_matrix.png",
    caption="Threshold applied: 0.33 (recall-precision balanced)",
)
add_footer(s)

# ---------------------------------------------------------------------------
# 8. Explainability
# ---------------------------------------------------------------------------
section_slide("Explainability", "Moving beyond 'the model said so'")

s = content_slide(
    "Global: Feature Importance (Permutation)",
    image_path=PLOTS / "feature_importance.png",
    caption="Higher importance = shuffling that feature degrades F1 the most",
)
add_footer(s)

s = slide()
add_bg(s)
title_bar(s, "Local: SHAP Explanations")
bullets(s, [
    ("What SHAP does:", "decomposes a prediction into per-feature contributions."),
    ("Formulation:", "P(attrition) = baseline risk + sum of feature contributions."),
    ("Interpretation:", "positive contribution pushes toward 'Leave', negative toward 'Stay'."),
    ("Why SVM:", "SVM probabilities are not inherently explainable - SHAP adds the explanation layer."),
    ("In the dashboard:", "each prediction shows the top 5 contributing risk factors."),
], 900000, 1300000, 10000000, 4000000)
tf3, _ = textbox(s, 900000, 5400000, 10000000, 800000)
p3 = tf3.paragraphs[0]
p3.text = "Example: Overtime=Yes (+0.11), DistanceFromHome (+0.10), StockOptionLevel (+0.06) ... -> HIGH RISK"
p3.font.size = 1300000
p3.font.color.rgb = ORANGE
p3.font.bold = True
add_footer(s)

# ---------------------------------------------------------------------------
# 9. Dashboard
# ---------------------------------------------------------------------------
section_slide("Interactive Dashboard", "Streamlit application with 4 pages")
s = slide()
add_bg(s)
title_bar(s, "Dashboard - Four Pages")
bullets(s, [
    ("1. Dashboard:", "KPIs (employees, attrition rate, best model), attrition distribution, department / overtime / job-role breakdowns."),
    ("2. Predict Attrition:", "enter 30 employee attributes -> probability, risk level (Low / Medium / High / Critical)."),
    ("3. Model Performance:", "comparison table, ROC curves, confusion matrix, F1 bar chart."),
    ("4. Explainability:", "global feature importance + SHAP methodology + top drivers table."),
    ("Engineering:", "models and data cached; explanation computed on demand for fast UX."),
], 900000, 1300000, 10000000, 4500000)
add_footer(s)

# ---------------------------------------------------------------------------
# 10. Conclusion & future work
# ---------------------------------------------------------------------------
s = slide()
add_bg(s)
title_bar(s, "Conclusion")
bullets(s, [
    ("Achieved:", "a complete, reproducible, explainable attrition classification system."),
    ("Best model:", "SVM with F1 = 0.559, ROC-AUC = 0.817 on unseen test data."),
    ("Robust evaluation:", "F1/ROC-AUC over accuracy due to an imbalanced target."),
    ("Explainable by design:", "permutation importance (global) + SHAP (local)."),
    ("Deployed:", "interactive Streamlit dashboard for non-technical HR users."),
], 900000, 1300000, 10000000, 4200000)
add_footer(s)

s = slide()
add_bg(s)
title_bar(s, "Future Work")
bullets(s, [
    ("SMOTE / cost-sensitive learning:", "explore advanced imbalance strategies if recall needs a further boost."),
    ("Model variety:", "XGBoost / LightGBM, and stacking of the top classifiers."),
    ("Time-aware data:", "use a longitudinal dataset to model tenure dynamics."),
    ("Fairness audit:", "check predictions across gender, age and department subgroups."),
    ("Deployment:", "containerise the dashboard and expose the model as an API."),
], 900000, 1300000, 10000000, 4500000)
add_footer(s)

# ---------------------------------------------------------------------------
# 11. Viva Q&A
# ---------------------------------------------------------------------------
s = slide()
add_bg(s)
title_bar(s, "Viva Q&A - Be Ready")
bullets(s, [
    ("Why classification?", "Attrition is binary (Yes/No) -> a binary classification problem."),
    ("Why multiple algorithms?", "each has different strengths; metrics decide the winner."),
    ("Why not only accuracy?", "16% positive class means a 'predict-all-stay' model scores 84% yet finds no leavers."),
    ("What does the model output?", "a probability of attrition; thresholded into a class label and risk level."),
    ("What is feature importance for?", "identifies which attributes most influence predictions (global)."),
    ("What is SHAP for?", "explains each single prediction (local) - e.g. why THIS employee is at risk."),
    ("Is it really ML?", "yes - patterns are learned from historical HR data, not hand-coded."),
], 900000, 1000000, 10000000, 5000000)
add_footer(s)

prs.save(str(PREZ_PATH))
print(f"Saved: {PREZ_PATH} ({PREZ_PATH.stat().st_size / 1024:.0f} KB, {len(prs.slides)} slides)")


def generate_phase_1_presentation():
    p1_path = ROOT / "reports" / "Phase_1_Presentation.pptx"
    p1 = Presentation()
    p1.slide_width = 12192000
    p1.slide_height = 6858000
    blank = p1.slide_layouts[6]

    def p1_slide():
        return p1.slides.add_slide(blank)

    def p1_footer(s):
        tf, _ = textbox(s, 300000, 6450000, p1.slide_width - 600000, 300000)
        p = tf.paragraphs[0]
        p.text = "East West Institute of Technology | Machine Learning [BIC703] | Phase 1 Evaluation"
        p.font.size = 900000
        p.font.color.rgb = GRAY
        p.alignment = PP_ALIGN.CENTER

    # 1. Title
    s = p1_slide()
    add_bg(s, NAVY)
    rect = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 1000000, p1.slide_width, 2400000)
    rect.fill.background()
    rect.line.fill.background()
    tf = rect.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "EAST WEST INSTITUTE OF TECHNOLOGY"
    p.font.size = 2200000
    p.font.bold = True
    p.font.color.rgb = ORANGE
    p.alignment = PP_ALIGN.CENTER

    p_d = tf.add_paragraph()
    p_d.text = "Department of CSE (IoT and Cyber Security including Blockchain Technology)\nAcademic Year 2026-27 | 7th Semester"
    p_d.font.size = 1300000
    p_d.font.color.rgb = WHITE
    p_d.alignment = PP_ALIGN.CENTER

    p_t = tf.add_paragraph()
    p_t.text = "\nEMPLOYEE ATTRITION PREDICTION USING ML"
    p_t.font.size = 3200000
    p_t.font.bold = True
    p_t.font.color.rgb = WHITE
    p_t.alignment = PP_ALIGN.CENTER

    p_sub = tf.add_paragraph()
    p_sub.text = "PHASE 1 EVALUATION PRESENTATION (40 MARKS)"
    p_sub.font.size = 1800000
    p_sub.font.color.rgb = ORANGE
    p_sub.alignment = PP_ALIGN.CENTER

    tf2, _ = textbox(s, 1500000, 4800000, 9200000, 1500000)
    p3 = tf2.paragraphs[0]
    p3.text = "Student Project Team | Machine Learning [BIC703]"
    p3.font.size = 1300000
    p3.font.color.rgb = WHITE
    p3.alignment = PP_ALIGN.CENTER

    p4 = tf2.add_paragraph()
    p4.text = "Course Instructor & Guide: Prof. P Bharath Krishna"
    p4.font.size = 1250000
    p4.font.color.rgb = ORANGE
    p4.alignment = PP_ALIGN.CENTER

    # 2. Phase 1 Agenda
    s = p1_slide()
    add_bg(s)
    title_bar(s, "Phase 1 Evaluation Agenda (Weeks 1 to 5)")
    bullets(s, [
        ("Week 1:", "Problem Identification, Domain Understanding & Objectives"),
        ("Week 2:", "Dataset Collection, Hygiene Assessment & Data Dictionary"),
        ("Week 3:", "Exploratory Data Analysis (EDA with 5+ Visualizations)"),
        ("Week 4:", "Data Cleaning, Preprocessing Pipeline & Leakage Prevention"),
        ("Week 5:", "Initial Machine Learning Models & Preliminary Benchmark"),
        ("Roadmap:", "Phase 2 Experimentation, Tuning, Explainability & Deployment"),
    ], 900000, 1300000, 10000000, 4500000)
    p1_footer(s)

    # 3. Problem Statement & Formulation
    s = p1_slide()
    add_bg(s)
    title_bar(s, "Problem Statement & Formulation")
    bullets(s, [
        ("Problem Statement:", "Can we proactively identify employees at risk of voluntary departure to mitigate organizational turnover costs?"),
        ("Formulation:", "Supervised Binary Classification; Target variable: Attrition ∈ {0: No, 1: Yes}."),
        ("Key Challenge:", "Extreme Class Imbalance (83.9% retained vs. 16.1% departed)."),
        ("Evaluation Focus:", "F1-Score, Precision, Recall and ROC-AUC rather than misleading raw accuracy."),
        ("Leakage Protection:", "All preprocessing encapsulated strictly inside scikit-learn Pipelines."),
    ], 900000, 1300000, 10000000, 4500000)
    p1_footer(s)

    # 4. Dataset & Data Dictionary
    s = p1_slide()
    add_bg(s)
    title_bar(s, "Dataset Overview & Data Hygiene")
    bullets(s, [
        ("Source:", "IBM HR Analytics Employee Attrition & Performance Benchmark."),
        ("Volume:", "1,470 employee records across 35 attributes."),
        ("Hygiene Audit:", "0 missing values across all columns; 0 duplicate records."),
        ("Cleaned Modeling Space:", "30 features retained (23 numerical + 7 categorical)."),
        ("Dropped Columns:", "EmployeeNumber (ID), EmployeeCount (constant), Over18 (constant), StandardHours (constant)."),
    ], 900000, 1300000, 10000000, 4500000)
    p1_footer(s)

    # 5. EDA Visualizations
    eda_dir = ROOT / "outputs" / "plots" / "eda"
    if (eda_dir / "eda_target_distribution.png").exists():
        s = content_slide("EDA 1: Target Class Imbalance", eda_dir / "eda_target_distribution.png", caption="Target distribution shows 83.9% retained vs 16.1% departed employees.")
        p1_footer(s)
    if (eda_dir / "eda_overtime_marital.png").exists():
        s = content_slide("EDA 2: OverTime & Marital Status Disparity", eda_dir / "eda_overtime_marital.png", caption="OverTime employees churn at ~30.5% vs ~10.4% non-overtime.")
        p1_footer(s)
    if (eda_dir / "eda_jobrole_attrition.png").exists():
        s = content_slide("EDA 3: Department & Job Role Churn", eda_dir / "eda_jobrole_attrition.png", caption="Sales Reps and Lab Techs suffer highest turnover.")
        p1_footer(s)
    if (eda_dir / "eda_satisfaction_trends.png").exists():
        s = content_slide("EDA 4: Satisfaction Rating Curves", eda_dir / "eda_satisfaction_trends.png", caption="Steep monotonic rise in attrition as satisfaction levels decline.")
        p1_footer(s)

    # 6. Preprocessing
    s = p1_slide()
    add_bg(s)
    title_bar(s, "Data Preprocessing & Pipeline Architecture")
    bullets(s, [
        ("Splitting:", "Stratified 70/30 train/test split preserving target class ratio."),
        ("Numerical Pipeline:", "StandardScaler() applied to 23 continuous and ordinal variables."),
        ("Categorical Pipeline:", "OneHotEncoder(handle_unknown='ignore') applied to 7 nominal variables."),
        ("Dimensionality:", "Expanded from 30 raw features to 51 orthogonal normalized dimensions."),
        ("Zero Leakage:", "ColumnTransformer fitted strictly on training folds inside cv loops."),
    ], 900000, 1300000, 10000000, 4500000)
    p1_footer(s)

    # 7. Initial Model Results
    s = p1_slide()
    add_bg(s)
    title_bar(s, "Initial Model Benchmark (Hold-Out Test Set)")
    bullets(s, [
        ("Support Vector Machine (Champion):", "F1 = 0.539 | ROC-AUC = 0.817 | Recall = 0.634 | Accuracy = 0.832"),
        ("Logistic Regression:", "F1 = 0.480 | ROC-AUC = 0.816 | Recall = 0.676 | Accuracy = 0.757"),
        ("Random Forest:", "F1 = 0.455 | ROC-AUC = 0.780 | Recall = 0.423 | Accuracy = 0.844"),
        ("Decision Tree:", "F1 = 0.414 | ROC-AUC = 0.698 | Recall = 0.577 | Accuracy = 0.741"),
        ("K-Nearest Neighbors:", "F1 = 0.286 | ROC-AUC = 0.611 | Recall = 0.197 | Accuracy = 0.841"),
        ("Key Insight:", "KNN has high accuracy (84.1%) but misses 80% of leavers, proving the Accuracy Paradox."),
    ], 900000, 1300000, 10000000, 4500000)
    p1_footer(s)

    # 8. Phase 2 Plan
    s = p1_slide()
    add_bg(s)
    title_bar(s, "Phase 2 Roadmap (Weeks 6 to 10)")
    bullets(s, [
        ("Week 6 - Deeper Experimentation:", "Explore ensemble behaviors and probability calibration."),
        ("Week 7 - Hyperparameter Tuning:", "5-fold StratifiedSearchCV across SVM, Random Forest and LR."),
        ("Week 8 - Threshold Tuning & Error Analysis:", "Precision-Recall optimization to maximize F1."),
        ("Week 9 - Web Deployment & GitHub:", "Streamlit enterprise dashboard with accelerated SHAP."),
        ("Week 10 - Final Report & Viva:", "Final 22-chapter thesis, presentation deck & viva defense."),
    ], 900000, 1300000, 10000000, 4500000)
    p1_footer(s)

    p1.save(str(p1_path))
    print(f"Saved Phase 1 Presentation: {p1_path} ({len(p1.slides)} slides)")


generate_phase_1_presentation()