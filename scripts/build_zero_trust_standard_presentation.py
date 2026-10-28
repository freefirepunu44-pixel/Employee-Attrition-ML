import os
from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "reports" / "assets_zt"
PLOTS_DIR = ROOT / "outputs" / "plots"
EDA_DIR = PLOTS_DIR / "eda"

VTU_LOGO = ASSETS_DIR / "image1.jpeg"
EWIT_LOGO = ASSETS_DIR / "image2.png"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
DARK = RGBColor(0x0F, 0x17, 0x2A)
MUTED = RGBColor(0x64, 0x74, 0x8B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ORANGE = RGBColor(0xF4, 0xA2, 0x59)
BLUE = RGBColor(0x25, 0x63, 0xEB)
CARD_BG = RGBColor(0xF8, 0xFA, 0xFC)
CARD_BORDER = RGBColor(0xCB, 0xD5, 0xE1)


def create_deck():
    print("Generating 15-Slide Presentation following Zero Trust presentation standards...")
    prs = Presentation()
    prs.slide_width = 12192000
    prs.slide_height = 6858000
    blank = prs.slide_layouts[6]

    def slide():
        return prs.slides.add_slide(blank)

    def header(s, num, title, subtitle):
        # Number badge
        badge = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, 450000, 350000, 520000, 520000)
        badge.fill.solid()
        badge.fill.fore_color.rgb = BLUE
        badge.line.fill.background()
        tf_b = badge.text_frame
        p_b = tf_b.paragraphs[0]
        p_b.text = num
        p_b.font.size = Pt(14)
        p_b.font.bold = True
        p_b.font.color.rgb = WHITE
        p_b.alignment = PP_ALIGN.CENTER

        # Title & Subtitle
        tx = s.shapes.add_textbox(1100000, 310000, 8500000, 650000)
        tf = tx.text_frame
        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.size = Pt(16.5)
        p1.font.bold = True
        p1.font.color.rgb = DARK

        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.font.size = Pt(11)
        p2.font.color.rgb = MUTED

        # Top-right banner
        tx_r = s.shapes.add_textbox(8000000, 320000, 3700000, 350000)
        tf_r = tx_r.text_frame
        p_r = tf_r.paragraphs[0]
        p_r.text = "EMPLOYEE ATTRITION ML  •  VTU PROJECT REVIEW"
        p_r.font.size = Pt(9.5)
        p_r.font.bold = True
        p_r.font.color.rgb = MUTED
        p_r.alignment = PP_ALIGN.RIGHT

    def card(s, x, y, w, h, title=None, body=None, bg=CARD_BG, border=CARD_BORDER):
        rect = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
        rect.fill.solid()
        rect.fill.fore_color.rgb = bg
        rect.line.color.rgb = border
        rect.line.width = Pt(1)
        tf = rect.text_frame
        tf.word_wrap = True
        tf.margin_left = 180000
        tf.margin_right = 180000
        tf.margin_top = 160000
        tf.margin_bottom = 160000

        if title:
            p0 = tf.paragraphs[0]
            p0.text = title
            p0.font.size = Pt(12)
            p0.font.bold = True
            p0.font.color.rgb = DARK
            p0.space_after = Pt(4)

        if body:
            p_start = tf.add_paragraph() if title else tf.paragraphs[0]
            p_start.text = body
            p_start.font.size = Pt(10)
            p_start.font.color.rgb = DARK
            p_start.space_after = Pt(2)

        return rect, tf

    # -------------------------------------------------------------
    # SLIDE 1: TITLE PAGE (Exact Match to Zero Trust Slide 1)
    # -------------------------------------------------------------
    s1 = slide()
    # Background
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = RGBColor(248, 250, 252)
    bg1.line.fill.background()

    # VTU & EWIT Headers
    tx_u = s1.shapes.add_textbox(1200000, 300000, 9792000, 700000)
    tf_u = tx_u.text_frame
    p_u1 = tf_u.paragraphs[0]
    p_u1.text = "VISVESVARAYA TECHNOLOGICAL UNIVERSITY"
    p_u1.font.size = Pt(16)
    p_u1.font.bold = True
    p_u1.font.color.rgb = DARK
    p_u1.alignment = PP_ALIGN.CENTER

    p_u2 = tf_u.add_paragraph()
    p_u2.text = "Jnana Sangama, Belgaum - 590014, Karnataka State, India"
    p_u2.font.size = Pt(10.5)
    p_u2.font.color.rgb = MUTED
    p_u2.alignment = PP_ALIGN.CENTER

    # Logos
    if VTU_LOGO.exists():
        s1.shapes.add_picture(str(VTU_LOGO), 500000, 260000, width=Inches(1.1))
    if EWIT_LOGO.exists():
        s1.shapes.add_picture(str(EWIT_LOGO), 10500000, 260000, width=Inches(1.1))

    tx_d = s1.shapes.add_textbox(1200000, 1100000, 9792000, 600000)
    tf_d = tx_d.text_frame
    p_d1 = tf_d.paragraphs[0]
    p_d1.text = "DEPARTMENT OF CSE (IoT & Cybersecurity including Blockchain)"
    p_d1.font.size = Pt(12)
    p_d1.font.bold = True
    p_d1.font.color.rgb = BLUE
    p_d1.alignment = PP_ALIGN.CENTER

    p_d2 = tf_d.add_paragraph()
    p_d2.text = "EAST WEST INSTITUTE OF TECHNOLOGY  •  Off Magadi Main Road, Bengaluru – 560 091"
    p_d2.font.size = Pt(10.5)
    p_d2.font.color.rgb = DARK
    p_d2.alignment = PP_ALIGN.CENTER

    # Center Hero Banner
    h_box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, 1200000, 1900000, 9792000, 1800000)
    h_box.fill.solid()
    h_box.fill.fore_color.rgb = NAVY
    h_box.line.fill.background()
    tf_h = h_box.text_frame
    tf_h.margin_top = 220000
    p_h0 = tf_h.paragraphs[0]
    p_h0.text = "MAJOR PROJECT PRESENTATION | AY 2025–2026"
    p_h0.font.size = Pt(11)
    p_h0.font.bold = True
    p_h0.font.color.rgb = ORANGE
    p_h0.alignment = PP_ALIGN.CENTER

    p_h1 = tf_h.add_paragraph()
    p_h1.text = "EMPLOYEE ATTRITION PREDICTION USING MACHINE LEARNING"
    p_h1.font.size = Pt(22)
    p_h1.font.bold = True
    p_h1.font.color.rgb = WHITE
    p_h1.alignment = PP_ALIGN.CENTER

    p_h2 = tf_h.add_paragraph()
    p_h2.text = "An Explainable Classification-Based Employee Risk Analysis System with Dual-Layer XAI"
    p_h2.font.size = Pt(12.5)
    p_h2.font.color.rgb = RGBColor(203, 213, 225)
    p_h2.alignment = PP_ALIGN.CENTER

    # Presented By & Guided By
    tx_p = s1.shapes.add_textbox(1500000, 4000000, 4500000, 2200000)
    tf_p = tx_p.text_frame
    pp0 = tf_p.paragraphs[0]
    pp0.text = "Presented By:"
    pp0.font.size = Pt(12)
    pp0.font.bold = True
    pp0.font.color.rgb = BLUE

    students_lines = [
        "ANUSHA H G : 1EW23IC006",
        "CHARAN M S : 1EW23IC012",
        "NITHIN GOWDA R : 1EW23IC031",
        "PUNARVA D N : 1EW23IC037",
    ]
    for s_line in students_lines:
        p_st = tf_p.add_paragraph()
        p_st.text = s_line
        p_st.font.size = Pt(11)
        p_st.font.bold = True
        p_st.font.color.rgb = DARK

    tx_g = s1.shapes.add_textbox(6800000, 4000000, 4500000, 2200000)
    tf_g = tx_g.text_frame
    pg0 = tf_g.paragraphs[0]
    pg0.text = "Under the Guidance of:"
    pg0.font.size = Pt(12)
    pg0.font.bold = True
    pg0.font.color.rgb = BLUE

    guide_lines = [
        "Dr. VENKATESHA G",
        "PROFESSOR & HOD",
        "Prof. P BHARATH KRISHNA",
        "Assistant Professor & Course Instructor",
        "Dept. of CSE (IoT & CyS Incl. BCT), EWIT",
    ]
    for g_line in guide_lines:
        p_gt = tf_g.add_paragraph()
        p_gt.text = g_line
        p_gt.font.size = Pt(10.5)
        p_gt.font.color.rgb = DARK
        if "Dr." in g_line or "Prof." in g_line:
            p_gt.font.bold = True

    # -------------------------------------------------------------
    # SLIDE 2: 01 | INTRODUCTION / BACKGROUND
    # -------------------------------------------------------------
    s2 = slide()
    header(s2, "01", "Introduction / Background", "Core principles of predictive people analytics and organizational turnover")
    card(s2, 450000, 1150000, 5400000, 4200000, "The Human Capital Crisis",
         "• The Hidden Cost: Voluntary turnover costs 50%–200% of an employee's annual salary in recruitment, onboarding, and productivity downtime.\n\n"
         "• Failure of Static HR: Conventional annual reviews and exit interviews are strictly post-facto—conducted after an employee resigns, offering zero retention window.\n\n"
         "• Why ML is Critical: Machine learning synthesizes 30+ multi-dimensional signals (overtime, commute distance, compensation, satisfaction) to predict departures weeks in advance.")

    card(s2, 6100000, 1150000, 5600000, 950000, "1. Leakage-Free Preprocessing",
         "scikit-learn ColumnTransformer scaling 23 numerical and encoding 7 categorical features strictly inside cross-validation loops.")
    card(s2, 6100000, 2200000, 5600000, 950000, "2. 5-Model Benchmark",
         "Evaluates SVM, Random Forest, Logistic Regression, Decision Tree, and KNN under stratified 5-fold cross-validation.")
    card(s2, 6100000, 3250000, 5600000, 950000, "3. Accuracy Paradox Resolution",
         "Tackles 84/16 class imbalance with cost-sensitive balanced loss weighting and optimal 0.33 decision threshold tuning.")
    card(s2, 6100000, 4300000, 5600000, 1050000, "4. Dual-Layer XAI & Web App",
         "Permutation Importance (global) + Accelerated SHAP (0.32s local) integrated into an enterprise Streamlit dashboard.")

    # -------------------------------------------------------------
    # SLIDE 3: 02 | LITERATURE REVIEW / RELATED WORK
    # -------------------------------------------------------------
    s3 = slide()
    header(s3, "02", "Literature Review / Related Work", "Analysis of existing talent analytics literature and identified research gaps")
    card(s3, 450000, 1150000, 11250000, 3800000, "Comparative Evaluation of Literature & Gap Analysis",
         "1. IBM Watson Analytics (2016): Published foundational 1,470-record HR dataset. Evaluated basic classifiers using raw accuracy, failing to catch rare leavers due to the Accuracy Paradox.\n\n"
         "2. Chawla et al. (2002) - SMOTE Oversampling: Proposed synthetic minority oversampling. In high-dimensional mixed tabular spaces, SMOTE generates noisy synthetic artifacts that corrupt decision boundaries.\n\n"
         "3. Breiman et al. (2001) - Random Forests: Proved bagging reduces variance, but MDI feature importance exhibits severe cardinality bias toward continuous features.\n\n"
         "4. Lundberg & Lee (2017) - SHAP (XAI): Formalized game-theoretic Shapley attributions. However, unoptimized KernelExplainer requires ~57s per sample on SVMs, precluding real-time web deployment.\n\n"
         "• PROJECT GAP ADDRESSED: Our system combines balanced cost-sensitive weighting, optimal threshold calibration (0.33), 170x accelerated SHAP (0.32s), and automated workforce batch CSV scoring into an end-to-end production tool.")

    # -------------------------------------------------------------
    # SLIDE 4: 03 | PROBLEM STATEMENT & MOTIVATION
    # -------------------------------------------------------------
    s4 = slide()
    header(s4, "03", "Problem Statement & Motivation", "The fatal flaws of reactive retention and severe economic costs")
    card(s4, 450000, 1150000, 11250000, 1000000, "Problem Statement",
         "“Traditional human resource management relies on reactive post-resignation exit surveys. Due to severe class imbalance (83.9% retention vs 16.1% attrition), standard models succumb to the Accuracy Paradox while complex black-box algorithms fail to provide actionable retention recommendations.”")
    card(s4, 450000, 2300000, 3550000, 3100000, "1. Severe Replacement Cost",
         "Departures cost $30,000–$75,000 per employee. Losing key technical leads destabilizes product schedules and destroys institutional domain memory.")
    card(s4, 4300000, 2300000, 3550000, 3100000, "2. The Accuracy Paradox",
         "A naive model predicting 'Stay' for all 1,470 records scores 83.9% accuracy with ZERO recall. Evaluating models on accuracy alone is disastrous in imbalanced domains.")
    card(s4, 8150000, 2300000, 3550000, 3100000, "3. Black-Box Dilemma",
         "HR managers cannot intervene without knowing WHY an employee is leaving. Explainable AI (XAI) is legally and operationally mandatory for enterprise adoption.")

    # -------------------------------------------------------------
    # SLIDE 5: 04 | OBJECTIVES & SCOPE
    # -------------------------------------------------------------
    s5 = slide()
    header(s5, "04", "Objectives & Scope", "Concrete measurable targets and defined project boundaries")
    card(s5, 450000, 1150000, 2100000, 2400000, "01. Extract", "Ingest 1,470 records, verify 0 missing values, drop 4 constants, and scale 30 features into 51 normalized inputs.")
    card(s5, 2700000, 1150000, 2100000, 2400000, "02. Train", "Benchmark 5 algorithms (SVM, RF, LR, DT, KNN) under 5-fold stratified cross-validation with balanced weighting.")
    card(s5, 4950000, 1150000, 2100000, 2400000, "03. Tune", "Execute GridSearchCV for hyperparameter selection and tune decision threshold from 0.50 to optimal 0.33.")
    card(s5, 7200000, 1150000, 2100000, 2400000, "04. Explain", "Accelerate SHAP KernelExplainer from 57s to 0.32s using k-means background centroids for real-time XAI.")
    card(s5, 9450000, 1150000, 2250000, 2400000, "05. Deploy", "Streamlit dashboard supporting single-employee presets and workforce batch CSV scoring with 1-click report export.")
    card(s5, 450000, 3750000, 11250000, 1700000, "Project Scope Boundaries",
         "• IN SCOPE: End-to-end ML pipeline • 30 features • 5 algorithms • F1 & ROC-AUC evaluation • Accelerated SHAP • Batch CSV scoring • 10 automated unit tests (100% pass).\n"
         "• DEVELOPMENT SCOPE & HONEST BOUNDARIES: Evaluated on IBM HR benchmark snapshot data; lacks longitudinal event-history intervals for continuous survival analysis. Production integration requires enterprise HRIS (Workday/SAP) connectors.")

    # -------------------------------------------------------------
    # SLIDE 6: 05 | PROPOSED METHODOLOGY / SYSTEM DESIGN
    # -------------------------------------------------------------
    s6 = slide()
    header(s6, "05", "Proposed Methodology / System Design", "4-Layer leak-free Machine Learning system architecture")
    card(s6, 450000, 1150000, 11250000, 950000, "1. DATA INGESTION & QUALITY AUDIT LAYER",
         "IBM HR Dataset (1,470 rows, 35 attributes)  •  Zero missing values  •  Zero duplicate records  •  Removal of EmployeeNumber, EmployeeCount, StandardHours, Over18")
    card(s6, 450000, 2200000, 11250000, 950000, "2. PREPROCESSING & PIPELINE LAYER (ZERO LEAKAGE)",
         "ColumnTransformer: StandardScaler on 23 numerical features + OneHotEncoder on 7 categoricals  •  Expands to 51 dense dimensions  •  Stratified 70/30 split")
    card(s6, 450000, 3250000, 11250000, 950000, "3. PREDICTIVE MODELING & TUNING LAYER",
         "Cost-Sensitive Balanced Loss Weighting  •  GridSearchCV 5-Fold Cross Validation  •  Threshold Optimization at 0.33  •  Champion Model: SVM (RBF Kernel)")
    card(s6, 450000, 4300000, 11250000, 1150000, "4. DUAL-LAYER EXPLAINABILITY & WEB DEPLOYMENT LAYER",
         "Global Permutation Importance  •  Accelerated Local SHAP (0.32s)  •  Streamlit Web Application  •  Batch Workforce CSV Scoring  •  Downloadable Scored Reports")

    # -------------------------------------------------------------
    # SLIDE 7: 05 | PROPOSED METHODOLOGY / WORKFLOW
    # -------------------------------------------------------------
    s7 = slide()
    header(s7, "05", "Proposed Methodology / Workflow", "Individual diagnostic and workforce batch assessment workflows")
    card(s7, 450000, 1150000, 5400000, 4300000, "INDIVIDUAL EMPLOYEE DIAGNOSTIC WORKFLOW",
         "• Step 1: Input Profile: User inputs 30 features or selects quick-fill Persona Preset (High-Risk Sales Rep, Low-Risk Senior Director, Average R&D).\n\n"
         "• Step 2: Pipeline Transformation: Input vector scaled and one-hot encoded using pre-fitted ColumnTransformer.\n\n"
         "• Step 3: Probability Inference: Champion SVM computes calibrated churn probability P(attrition).\n\n"
         "• Step 4: Decision & Risk Tiering: Threshold 0.33 classifies verdict; mapped to Low (<30%), Medium (30-55%), High (55-80%), Critical (≥80%).\n\n"
         "• Step 5: Instant SHAP Explanation: Computes top-6 risk factors in 0.32s; renders interactive waterfall attribution bar chart.")

    card(s7, 6100000, 1150000, 5600000, 4300000, "WORKFORCE BATCH CSV SCORING WORKFLOW",
         "• Step 1: Roster Upload: HR executive uploads corporate employee CSV roster or loads 100 sample records.\n\n"
         "• Step 2: Vectorized Batch Inference: Ingests full dataset and scores all records in parallel (<100 ms).\n\n"
         "• Step 3: Workforce Analytics: Computes predicted department attrition rates, risk tier breakdown donut chart, and average churn probability.\n\n"
         "• Step 4: High-Risk Ranking: Ranks employees by churn risk, highlighting intervention priorities.\n\n"
         "• Step 5: Exportable Scored Report: One-click export of employee_attrition_scored.csv containing predictions, probabilities, and risk tiers.")

    # -------------------------------------------------------------
    # SLIDE 8: 06 | TOOLS, TECHNOLOGIES & IMPLEMENTATION
    # -------------------------------------------------------------
    s8 = slide()
    header(s8, "06", "Tools, Technologies & Implementation", "Technology stack and engineering components powering the system")
    card(s8, 450000, 1150000, 5400000, 2050000, "Core Platform & Machine Learning",
         "• Language: Python 3.14.3 with strict type annotations\n"
         "• ML Framework: scikit-learn 1.5 & Joblib 1.3\n"
         "• Algorithms: Support Vector Classifier (RBF), Random Forest, Logistic Regression, Decision Tree, KNN\n"
         "• Hyperparameter Tuning: Stratified 5-Fold GridSearchCV")
    card(s8, 6100000, 1150000, 5600000, 2050000, "Explainability (XAI) & Analytics",
         "• Local Explainability: SHAP 0.45 KernelExplainer & TreeExplainer\n"
         "• 170x Acceleration: shap.kmeans(background, 10) + explainer caching\n"
         "• Global Importance: Model-agnostic Permutation Importance\n"
         "• Mathematical Rigor: Game-theoretic Shapley value attribution")
    card(s8, 450000, 3350000, 5400000, 2100000, "Interactive Web Application",
         "• Dashboard Framework: Streamlit 1.36 Glassmorphic UI\n"
         "• Dynamic Visualizations: Plotly 5.20 Graph Objects\n"
         "• Presets: Pre-configured employee personas for instant testing\n"
         "• Batch Processing: Vectorized multi-record CSV scoring engine")
    card(s8, 6100000, 3350000, 5600000, 2100000, "Quality Assurance & Deliverables",
         "• Unit Testing: Python unittest suite (10/10 passing tests in 4.88s)\n"
         "• Documentation: python-docx 1.1 generating 22-chapter thesis\n"
         "• Presentation: python-pptx 1.0 generating 15-slide deck\n"
         "• Measured Latency: Single predict ~12ms | SHAP ~320ms")

    # -------------------------------------------------------------
    # SLIDE 9: 07 | RESULTS & DISCUSSION
    # -------------------------------------------------------------
    s9 = slide()
    header(s9, "07", "Results & Discussion", "Empirical benchmark comparison on 30% hold-out test set (441 employees)")
    card(s9, 450000, 1150000, 11250000, 2500000, "Hold-Out Benchmark Across 5 Classifiers",
         "• Support Vector Machine (Champion): F1 = 0.539 | ROC-AUC = 0.817 | Recall = 0.634 | Precision = 0.474 | CV F1 = 0.553\n"
         "• Logistic Regression: F1 = 0.480 | ROC-AUC = 0.816 | Recall = 0.676 | Precision = 0.366 | CV F1 = 0.512\n"
         "• Random Forest: F1 = 0.455 | ROC-AUC = 0.780 | Recall = 0.423 | Precision = 0.488 | CV F1 = 0.514\n"
         "• Decision Tree: F1 = 0.414 | ROC-AUC = 0.698 | Recall = 0.577 | Precision = 0.320 | CV F1 = 0.408\n"
         "• K-Nearest Neighbors: F1 = 0.286 | ROC-AUC = 0.611 | Recall = 0.197 | Precision = 0.519 | CV F1 = 0.284\n\n"
         "• THE ACCURACY PARADOX: KNN scored high accuracy (84.1%) but lowest F1 (0.286), missing 80% of leavers!")

    card(s9, 450000, 3800000, 5400000, 1650000, "Optimal Threshold Calibration (0.33)",
         "Lowering decision threshold from 0.50 to 0.33 raised SVM test F1 to 0.5594, Precision to 0.5556, Recall to 0.5634, and Accuracy to 0.8571.")
    card(s9, 6100000, 3800000, 5600000, 1650000, "Economic Cost Matrix (FN vs FP)",
         "Caught 40 true leavers (saving ~$1.2M in turnover costs). False Positives (32) cost only ~$1,000 each in check-ins. Cost(FN) >> Cost(FP).")

    # -------------------------------------------------------------
    # SLIDE 10: 08 | TESTING & VALIDATION
    # -------------------------------------------------------------
    s10 = slide()
    header(s10, "08", "Testing & Validation", "Verification of preprocessing, inference, probability bounds, and metrics")
    card(s10, 450000, 1150000, 2600000, 2400000, "01. Data & Preprocessing", "test_load_data_shape_and_columns: PASS\ntest_prepare_dataframe: PASS\ntest_preprocessor_transformation: PASS (51 dense features)")
    card(s10, 3250000, 1150000, 2600000, 2400000, "02. Prediction Service", "test_load_model_bundle_keys: PASS\ntest_predict_probability_bounds: PASS (0 <= P <= 1)\ntest_risk_level_mapping: PASS")
    card(s10, 6050000, 1150000, 2600000, 2400000, "03. XAI & Batch Scoring", "test_predict_with_explanation: PASS (Instant SHAP ranked attributions)\ntest_batch_predict: PASS (DataFrame multi-record scoring)")
    card(s10, 8850000, 1150000, 2850000, 2400000, "04. Metrics & Threshold", "test_compute_metrics: PASS\ntest_compute_threshold_curve: PASS\nTotal: 10/10 OK (4.88s)")
    card(s10, 450000, 3750000, 11250000, 1700000, "Scope of Validation & Honest Boundaries",
         "• Rigorous Automated Testing: All pipeline contracts, dimensional transformations, and probability bounds validated via unittest suite.\n"
         "• Development Scope Disclosures: Evaluated on IBM HR benchmark snapshot data; production deployment requires enterprise database connectors and scheduled re-training pipelines.")

    # -------------------------------------------------------------
    # SLIDE 11: 09 | RISKS, CHALLENGES & MITIGATION
    # -------------------------------------------------------------
    s11 = slide()
    header(s11, "09", "Risks, Challenges & Mitigation", "Identified technical challenges and operational countermeasures")
    card(s11, 450000, 1150000, 5400000, 2050000, "1. Severe Class Imbalance (84:16)",
         "• Challenge: Models default to predicting 'Stay', scoring high accuracy while finding zero leavers.\n"
         "• Countermeasure: Applied cost-sensitive balanced loss weighting (w_1=3.101, w_0=0.596) and tuned threshold to 0.33.")
    card(s11, 6100000, 1150000, 5600000, 2050000, "2. SHAP Inference Latency (57 seconds)",
         "• Challenge: KernelExplainer evaluating on 100 raw background rows took 57 seconds per prediction.\n"
         "• Countermeasure: Summarized background to 10 weighted k-means centroids, cutting runtime to 0.32 seconds.")
    card(s11, 450000, 3350000, 5400000, 2100000, "3. Data Leakage Vulnerability",
         "• Challenge: Preprocessing transformers fitted on full dataset leak test set statistics.\n"
         "• Countermeasure: Encapsulated ColumnTransformer strictly inside scikit-learn Pipelines with 5-fold CV.")
    card(s11, 6100000, 3350000, 5600000, 2100000, "4. Algorithmic Fairness & Bias",
         "• Challenge: Features such as Gender and Marital Status can introduce discriminatory bias.\n"
         "• Countermeasure: Permutation importance proves tenure, overtime, and satisfaction dominate predictions.")

    # -------------------------------------------------------------
    # SLIDE 12: 10 | APPLICATIONS, FUTURE SCOPE & CONCLUSION
    # -------------------------------------------------------------
    s12 = slide()
    header(s12, "10", "Applications, Future Scope & Conclusion", "Significance, deployment scope, and strategic roadmap")
    card(s12, 450000, 1150000, 3550000, 4300000, "Enterprise Applications",
         "• Engineering & IT: Flag overworked software developers facing burnout.\n\n"
         "• Sales Organizations: Prevent churn in high-turnover sales representative roles.\n\n"
         "• Healthcare Systems: Support hospital staffing resilience by monitoring nurse and physician workload.\n\n"
         "• Corporate Compensation: Optimize stock option allocation to retain top technical talent.")
    card(s12, 4300000, 1150000, 3550000, 4300000, "Future Scope Roadmap",
         "• Enterprise HRIS Connectors: Build secure REST APIs connecting directly to Workday and SAP SuccessFactors.\n\n"
         "• NLP Sentiment Extraction: Ingest free-text annual review comments using transformer embeddings.\n\n"
         "• Deep Survival Analysis: Deploy Cox Proportional Hazards modeling to predict exact departure months.\n\n"
         "• Fairness Audits: Automated monitoring of demographic parity across protected subgroups.")
    card(s12, 8150000, 1150000, 3550000, 4300000, "Project Conclusion",
         "• System Validated: Successfully developed an authentic, explainable employee attrition prediction platform.\n\n"
         "• Champion Performance: SVM achieves test F1 = 0.5594, ROC-AUC = 0.8168 at optimal threshold 0.33.\n\n"
         "• Real-Time XAI: 170x accelerated SHAP delivers actionable factor attributions in 0.32 seconds.\n\n"
         "• Production Ready: Enterprise Streamlit application with single and batch workforce scoring.")

    # -------------------------------------------------------------
    # SLIDE 13: 11 | LINK TO RELEVANT SDGS
    # -------------------------------------------------------------
    s13 = slide()
    header(s13, "11", "Link to Relevant SDGs", "Alignment with UN Sustainable Development Goals for decent work and economic resilience")
    card(s13, 450000, 1150000, 3550000, 4300000, "SDG 8: Decent Work & Economic Growth",
         "Target Alignment: Target 8.5 & Target 8.8: Achieve productive employment and protect labour rights.\n\n"
         "Workplace Well-Being: Excessive overtime and poor work-life balance are primary causes of employee burnout and voluntary turnover.\n\n"
         "Project Contribution: By diagnosing overtime stress and low satisfaction early, our system enables HR leadership to rebalance workloads and foster healthier working environments.")

    card(s13, 4300000, 1150000, 3550000, 4300000, "SDG 9: Industry, Innovation & Infrastructure",
         "Target Alignment: Target 9.4 & Target 9.5: Upgrade enterprise technological capabilities.\n\n"
         "Operational Resilience: High employee turnover disrupts manufacturing and technical R&D pipelines.\n\n"
         "Project Contribution: Deploying predictive machine learning transforms human resources from reactive administrative processing into an intelligent, data-driven management system.")

    card(s13, 8150000, 1150000, 3550000, 4300000, "SDG 10: Reduced Inequalities",
         "Target Alignment: Target 10.2 & Target 10.4: Promote equitable treatment and eliminate compensation disparity.\n\n"
         "Fairness & Transparency: Black-box retention decisions risk unfair managerial favoritism.\n\n"
         "Project Contribution: Our Dual-Layer XAI architecture ensures that retention interventions are based on transparent, audited feature attributions rather than subjective bias.")

    # -------------------------------------------------------------
    # SLIDE 14: 12 | REFERENCES
    # -------------------------------------------------------------
    s14 = slide()
    header(s14, "12", "References", "Authoritative IEEE standards and publications in Machine Learning and XAI")
    card(s14, 450000, 1150000, 11250000, 4300000, "Cited Literature",
         "1. S. M. Lundberg and S. I. Lee, “A unified approach to interpreting model predictions,” in Advances in Neural Information Processing Systems (NeurIPS 2017), vol. 30, pp. 4765–4774, 2017.\n\n"
         "2. F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, Nov. 2011.\n\n"
         "3. IBM Watson Analytics, “HR Employee Attrition and Performance Dataset,” IBM Corporation, Kaggle Open Datasets, 2016.\n\n"
         "4. N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: synthetic minority over-sampling technique,” Journal of Artificial Intelligence Research, vol. 16, pp. 321–357, Jun. 2002.\n\n"
         "5. L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, Oct. 2001. DOI: 10.1023/A:1010933404324.\n\n"
         "6. C. Cortes and V. Vapnik, “Support-vector networks,” Machine Learning, vol. 20, no. 3, pp. 273–297, Sep. 1995. DOI: 10.1007/BF00994018.\n\n"
         "7. Society for Human Resource Management (SHRM), “Human Capital Benchmarking Report,” SHRM Research, Alexandria, VA, USA, Tech. Rep., 2023.")

    # -------------------------------------------------------------
    # SLIDE 15: 13 | THANK YOU & VIVA DEFENSE
    # -------------------------------------------------------------
    s15 = slide()
    bg15 = s15.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg15.fill.solid()
    bg15.fill.fore_color.rgb = NAVY
    bg15.line.fill.background()

    # Badge
    b15 = s15.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, 5500000, 1200000, 1192000, 1192000)
    b15.fill.solid()
    b15.fill.fore_color.rgb = BLUE
    b15.line.fill.background()
    tf_b15 = b15.text_frame
    p_b15 = tf_b15.paragraphs[0]
    p_b15.text = "AI"
    p_b15.font.size = Pt(28)
    p_b15.font.bold = True
    p_b15.font.color.rgb = WHITE
    p_b15.alignment = PP_ALIGN.CENTER

    tx_ty = s15.shapes.add_textbox(1500000, 2600000, 9192000, 1800000)
    tf_ty = tx_ty.text_frame
    p_ty1 = tf_ty.paragraphs[0]
    p_ty1.text = "Thank You"
    p_ty1.font.size = Pt(36)
    p_ty1.font.bold = True
    p_ty1.font.color.rgb = WHITE
    p_ty1.alignment = PP_ALIGN.CENTER

    p_ty2 = tf_ty.add_paragraph()
    p_ty2.text = "Employee Attrition Prediction Using Machine Learning  •  Explainable AI with Fast SHAP"
    p_ty2.font.size = Pt(14)
    p_ty2.font.color.rgb = ORANGE
    p_ty2.alignment = PP_ALIGN.CENTER

    p_ty3 = tf_ty.add_paragraph()
    p_ty3.text = "\nQuestions & Discussion | Project Viva Defense"
    p_ty3.font.size = Pt(16)
    p_ty3.font.bold = True
    p_ty3.font.color.rgb = WHITE
    p_ty3.alignment = PP_ALIGN.CENTER

    tx_st = s15.shapes.add_textbox(1500000, 4800000, 9192000, 1200000)
    tf_st = tx_st.text_frame
    p_st1 = tf_st.paragraphs[0]
    p_st1.text = "ANUSHA H G : 1EW23IC006       •       CHARAN M S : 1EW23IC012"
    p_st1.font.size = Pt(12)
    p_st1.font.bold = True
    p_st1.font.color.rgb = RGBColor(203, 213, 225)
    p_st1.alignment = PP_ALIGN.CENTER

    p_st2 = tf_st.add_paragraph()
    p_st2.text = "NITHIN GOWDA R : 1EW23IC031       •       PUNARVA D N : 1EW23IC037"
    p_st2.font.size = Pt(12)
    p_st2.font.bold = True
    p_st2.font.color.rgb = RGBColor(203, 213, 225)
    p_st2.alignment = PP_ALIGN.CENTER

    p_guide_c = tf_st.add_paragraph()
    p_guide_c.text = "Under the Guidance of: Prof. P Bharath Krishna & Dr. Venkatesha G (HOD)"
    p_guide_c.font.size = Pt(11)
    p_guide_c.font.color.rgb = ORANGE
    p_guide_c.alignment = PP_ALIGN.CENTER

    out1 = ROOT / "Employee_Attrition_Presentation.pptx"
    out2 = ROOT / "reports" / "Employee_Attrition_VTU_Project_Presentation.pptx"
    prs.save(str(out1))
    prs.save(str(out2))
    print(f"Successfully generated:\n- {out1}\n- {out2}")


if __name__ == "__main__":
    create_deck()
