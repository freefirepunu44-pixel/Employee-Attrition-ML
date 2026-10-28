import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict import (
    batch_predict,
    explain_prediction,
    load_model,
    predict_probability,
    risk_level,
)
from preprocessing import FEATURES, load_data

INK = "#0F172A"
MUT = "#64748B"
FAINT = "#E2E8F0"
SURFACE = "#FFFFFF"
BG = "#F8FAFC"
PRIMARY = "#2563EB"
PRIMARY_DK = "#1E40AF"
EMERALD = "#10B981"
RED = "#EF4444"
AMBER = "#F59E0B"
VIOLET = "#8B5CF6"

RISK = {"Low": EMERALD, "Medium": AMBER, "High": RED, "Critical": "#7F1D1D"}

PRESETS = {
    "🚨 High Risk: Sales Executive (Overtime, Single, High Churn Profile)": {
        "Age": 41, "BusinessTravel": "Travel_Rarely", "DailyRate": 1102, "Department": "Sales",
        "DistanceFromHome": 8, "Education": 2, "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 2, "Gender": "Female", "HourlyRate": 94,
        "JobInvolvement": 3, "JobLevel": 2, "JobRole": "Sales Executive",
        "JobSatisfaction": 2, "MaritalStatus": "Single", "MonthlyIncome": 4800,
        "MonthlyRate": 19479, "NumCompaniesWorked": 8, "OverTime": "Yes",
        "PercentSalaryHike": 11, "PerformanceRating": 3,
        "RelationshipSatisfaction": 1, "StockOptionLevel": 0,
        "TotalWorkingYears": 8, "TrainingTimesLastYear": 1,
        "WorkLifeBalance": 1, "YearsAtCompany": 4,
        "YearsInCurrentRole": 2, "YearsSinceLastPromotion": 2,
        "YearsWithCurrManager": 2,
    },
    "🔥 Extreme Risk: Sales Rep (Frequent Travel, Low Pay, Overtime)": {
        "Age": 28, "BusinessTravel": "Travel_Frequently", "DailyRate": 600, "Department": "Sales",
        "DistanceFromHome": 24, "Education": 2, "EducationField": "Marketing",
        "EnvironmentSatisfaction": 1, "Gender": "Male", "HourlyRate": 55,
        "JobInvolvement": 2, "JobLevel": 1, "JobRole": "Sales Representative",
        "JobSatisfaction": 1, "MaritalStatus": "Single", "MonthlyIncome": 2400,
        "MonthlyRate": 14000, "NumCompaniesWorked": 4, "OverTime": "Yes",
        "PercentSalaryHike": 11, "PerformanceRating": 3,
        "RelationshipSatisfaction": 1, "StockOptionLevel": 0,
        "TotalWorkingYears": 3, "TrainingTimesLastYear": 1,
        "WorkLifeBalance": 1, "YearsAtCompany": 2,
        "YearsInCurrentRole": 1, "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 1,
    },
    "⚠️ Moderate Risk: Laboratory Technician (Modest Pay, Stagnant Role)": {
        "Age": 29, "BusinessTravel": "Travel_Rarely", "DailyRate": 591, "Department": "Research & Development",
        "DistanceFromHome": 9, "Education": 2, "EducationField": "Medical",
        "EnvironmentSatisfaction": 2, "Gender": "Male", "HourlyRate": 45,
        "JobInvolvement": 2, "JobLevel": 1, "JobRole": "Laboratory Technician",
        "JobSatisfaction": 2, "MaritalStatus": "Single", "MonthlyIncome": 3200,
        "MonthlyRate": 16632, "NumCompaniesWorked": 4, "OverTime": "No",
        "PercentSalaryHike": 12, "PerformanceRating": 3,
        "RelationshipSatisfaction": 3, "StockOptionLevel": 0,
        "TotalWorkingYears": 6, "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 2, "YearsAtCompany": 3,
        "YearsInCurrentRole": 2, "YearsSinceLastPromotion": 2,
        "YearsWithCurrManager": 2,
    },
    "🟢 Low Risk: Senior Manufacturing Director (High Tenure & Income)": {
        "Age": 46, "BusinessTravel": "Travel_Rarely", "DailyRate": 900, "Department": "Research & Development",
        "DistanceFromHome": 3, "Education": 4, "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 4, "Gender": "Female", "HourlyRate": 78,
        "JobInvolvement": 4, "JobLevel": 3, "JobRole": "Manufacturing Director",
        "JobSatisfaction": 4, "MaritalStatus": "Married", "MonthlyIncome": 12800,
        "MonthlyRate": 16000, "NumCompaniesWorked": 1, "OverTime": "No",
        "PercentSalaryHike": 18, "PerformanceRating": 3,
        "RelationshipSatisfaction": 4, "StockOptionLevel": 2,
        "TotalWorkingYears": 18, "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 3, "YearsAtCompany": 12,
        "YearsInCurrentRole": 8, "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 7,
    },
    "✏️ Custom Employee Profile (Enter Details Below)": None,
}

st.set_page_config(
    page_title="Employee Attrition Predictor | AI Decision Support",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700&display=swap');

html, body, .stApp, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  color: #0F172A;
}
.stApp { background: #F8FAFC; }
.block-container { padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1260px; }

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0A192F 0%, #112240 60%, #1B3054 100%);
  border-right: 1px solid rgba(255,255,255,.07);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
div[role="radiogroup"] { gap: 6px; }
div[role="radiogroup"] label {
  padding: 0.65rem 0.85rem !important; border-radius: 10px; cursor: pointer;
  transition: all .15s ease;
}
div[role="radiogroup"] label:hover { background: rgba(255,255,255,.08); }
div[role="radiogroup"] label:has(input:checked) {
  background: rgba(37,99,235,.35);
  box-shadow: inset 0 0 0 1px rgba(96,165,250,.55);
}
div[role="radiogroup"] p, div[role="radiogroup"] [data-testid="stMarkdownContainer"] p {
  font-weight: 600 !important; font-size: 13.5px !important; color: #E2E8F0 !important;
}

/* ---------- Buttons ---------- */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius: 10px; font-weight: 700; border: none;
  background: linear-gradient(135deg,#2563EB,#1E40AF); color: #fff;
  padding: 0.68em 1.5em; transition: all .18s; box-shadow: 0 4px 14px rgba(37,99,235,.28);
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
  transform: translateY(-1px); box-shadow: 0 8px 22px rgba(37,99,235,.38); color: #fff;
}

/* ---------- Cards & Panels ---------- */
.panel {
  background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px;
  padding: 22px 24px; box-shadow: 0 4px 18px rgba(15,23,42,.04);
  margin-bottom: 20px;
}
.panel-title { font-size: 16px; font-weight: 800; color: #0F172A; }
.panel-sub { font-size: 12.8px; color: #64748B; margin-top: 3px; }

.kpi {
  background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:18px 20px;
  box-shadow:0 4px 16px rgba(15,23,42,.04); position:relative; overflow:hidden;
}
.kpi .accent { position:absolute; left:0; top:0; bottom:0; width:5px; }
.kpi .icon {
  width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center;
  font-size:18px; margin-bottom:10px;
}
.kpi .lbl { font-size:11px; font-weight:700; letter-spacing:.07em; text-transform:uppercase; color:#64748B; }
.kpi .val { font-size:26px; font-weight:900; color:#0F172A; margin-top:2px; letter-spacing:-.01em; }
.kpi .sub { font-size:12px; color:#94A3B8; margin-top:3px; }

.verdict-box {
  border-radius: 16px; padding: 22px 26px; margin-bottom: 20px;
  border-width: 2px; border-style: solid;
}
.verdict-box.high {
  background: #FEF2F2; border-color: #F87171;
}
.verdict-box.low {
  background: #F0FDF4; border-color: #4ADE80;
}

.pill {
  display:inline-flex; align-items:center; gap:6px; padding:6px 16px; border-radius:999px;
  font-size:13px; font-weight:800; text-transform:uppercase; letter-spacing:.04em;
}
.tag {
  display:inline-block; padding:3px 10px; border-radius:8px; font-size:11.5px; font-weight:700;
  letter-spacing:.05em; text-transform:uppercase; background:#E8EFFA; color:#1E40AF;
}
.section-label { font-size:12.5px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:#64748B; margin:22px 0 12px; }

.gauge-track { display:flex; gap:6px; margin-top:12px; }
.gauge-seg { flex:1; height:14px; border-radius:4px; background:#E2E8F0; transition: background .4s; }

.mono { font-family:'JetBrains Mono', monospace; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_model()


@st.cache_data(show_spinner=False)
def get_data():
    return load_data()


@st.cache_data(show_spinner=False)
def get_results():
    with open(ROOT / "outputs" / "model_results.json", "r", encoding="utf-8") as f:
        return json.load(f)


def short_model(name):
    return {
        "Support Vector Machine": "SVM (Tuned RBF)",
        "Logistic Regression": "Logistic Regression",
        "Random Forest": "Random Forest",
        "Decision Tree": "Decision Tree",
        "K-Nearest Neighbors": "K-Nearest Neighbors",
    }.get(name, name)


def style_fig(fig, height=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=INK, size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        hoverlabel=dict(font=dict(family="Inter", size=12)),
    )
    fig.update_xaxes(gridcolor="#E8EEF5", zeroline=False, showline=False, tickfont=dict(color=MUT))
    fig.update_yaxes(gridcolor="#E8EEF5", zeroline=False, tickfont=dict(color=MUT))
    if height:
        fig.update_layout(height=height)
    return fig


def kpi_card(name, val, sub, icon, accent, icon_bg):
    return f"""
    <div class="kpi">
      <div class="accent" style="background:{accent};"></div>
      <div class="icon" style="background:{icon_bg};">{icon}</div>
      <div class="lbl">{name}</div>
      <div class="val">{val}</div>
      <div class="sub">{sub}</div>
    </div>"""


def seg_gauge(fraction, color):
    segs = 12
    filled = int(round(max(0.0, min(fraction, 1.0)) * segs))
    cells = "".join(
        f'<div class="gauge-seg" style="background:{color if i < filled else "#E2E8F0"};"></div>'
        for i in range(segs)
    )
    return f'<div class="gauge-track">{cells}</div>'


def sidebar_navigation():
    st.sidebar.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;padding:4px 6px 20px;">
          <div style="width:42px;height:42px;border-radius:12px;
                      background:linear-gradient(135deg,#2563EB,#8B5CF6);
                      display:flex;align-items:center;justify-content:center;
                      font-weight:900;color:#fff;font-size:18px;">👥</div>
          <div>
            <div style="font-size:17px;font-weight:900;color:#fff;letter-spacing:-.01em;">AttritionIQ</div>
            <div style="font-size:11.5px;color:#94B3D6;">Attrition Prediction System</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "🎯 Predict Employee Attrition",
            "📁 Batch Workforce CSV Predictor",
            "🔄 Retention Intervention Simulator",
            "📊 Workforce Historical Data & EDA",
            "🧠 Model Performance & Validation",
        ],
        index=0,
    )

    st.sidebar.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div style="font-size:11px;font-weight:800;letter-spacing:.08em;'
        'text-transform:uppercase;color:#8FA8C8;margin-bottom:8px;">Champion ML Engine</div>',
        unsafe_allow_html=True,
    )
    b = get_bundle()
    st.sidebar.markdown(
        f"""
        <div style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.14);
                    border-radius:14px;padding:14px 16px;">
          <div style="font-size:13px;color:#fff;font-weight:800;">Support Vector Machine</div>
          <div style="font-size:11.5px;color:#94B3D6;margin-top:2px;">RBF Kernel · Balanced Weights</div>
          <div style="display:flex;gap:16px;margin-top:12px;">
            <div>
              <div style="font-size:10px;color:#94A3B8;font-weight:700;">TEST F1</div>
              <div style="font-size:16px;font-weight:900;color:#fff;">{b['metrics']['f1']:.2f}</div>
            </div>
            <div>
              <div style="font-size:10px;color:#94A3B8;font-weight:700;">ROC-AUC</div>
              <div style="font-size:16px;font-weight:900;color:#fff;">{b['metrics']['roc_auc']:.2f}</div>
            </div>
            <div>
              <div style="font-size:10px;color:#94A3B8;font-weight:700;">THRESHOLD</div>
              <div style="font-size:16px;font-weight:900;color:#60A5FA;">{b['threshold']:.2f}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        f"""
        <div style="position:fixed;bottom:16px;left:20px;right:20px;font-size:11px;color:#64748B;line-height:1.6;">
          East West Institute of Technology<br>VTU Final Year Capstone Project
        </div>
        """,
        unsafe_allow_html=True,
    )
    return page


# ===========================================================================
# PAGE 1: PREDICT EMPLOYEE ATTRITION (PRIMARY DEFAULT VIEW)
# ===========================================================================
def render_predict_page():
    b = get_bundle()
    threshold = float(b["threshold"])

    st.markdown(
        """
        <div style="margin-bottom:18px;">
          <div style="font-size:30px;font-weight:900;color:#0F172A;letter-spacing:-.02em;">
            Employee Attrition Risk Predictor
          </div>
          <div style="font-size:14.5px;color:#64748B;margin-top:4px;">
            Predict whether an employee will stay or leave using the calibrated Support Vector Machine model.
            Get real-time probability, risk classification, explainable SHAP drivers, and actionable retention recommendations.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Preset selector bar
    st.markdown(
        '<div style="font-size:12px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#64748B;margin-bottom:6px;">'
        'Quick-Load Employee Persona</div>',
        unsafe_allow_html=True,
    )
    selected_preset_key = st.selectbox(
        "Load Preset Profile",
        list(PRESETS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    preset_vals = PRESETS.get(selected_preset_key)

    def val(k, default):
        if preset_vals and k in preset_vals:
            return preset_vals[k]
        return default

    with st.form("employee_prediction_form"):
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">'
            '<span class="tag">Section 1</span><span style="font-weight:800;font-size:15px;color:#0F172A;">Demographics & Commute</span></div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 65, int(val("Age", 35)))
        gender_opts = ["Male", "Female"]
        gender = c1.selectbox("Gender", gender_opts, index=gender_opts.index(val("Gender", "Male")))
        mar_opts = ["Single", "Married", "Divorced"]
        marital = c2.selectbox("Marital Status", mar_opts, index=mar_opts.index(val("MaritalStatus", "Single")))
        distance = c2.number_input("Distance From Home (Miles)", 1, 35, int(val("DistanceFromHome", 8)))
        edu_opts = ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]
        edu_field = c3.selectbox("Education Field", edu_opts, index=edu_opts.index(val("EducationField", "Life Sciences")))
        education = c3.slider("Education Level (1=Below College, 4=Master, 5=Doctor)", 1, 5, int(val("Education", 3)))

        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin:20px 0 12px;">'
            '<span class="tag">Section 2</span><span style="font-weight:800;font-size:15px;color:#0F172A;">Job Role, Department & Compensation</span></div>',
            unsafe_allow_html=True,
        )
        c4, c5, c6 = st.columns(3)
        role_opts = [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources",
        ]
        role = c4.selectbox("Job Role", role_opts, index=role_opts.index(val("JobRole", "Sales Executive")))
        dept_opts = ["Sales", "Research & Development", "Human Resources"]
        dept = c4.selectbox("Department", dept_opts, index=dept_opts.index(val("Department", "Sales")))
        job_level = c4.slider("Job Level (1 to 5)", 1, 5, int(val("JobLevel", 2)))

        income = c5.number_input("Monthly Income (USD $)", 1000, 25000, int(val("MonthlyIncome", 4800)), step=100)
        hike = c5.number_input("Percent Salary Hike (%)", 10, 30, int(val("PercentSalaryHike", 11)))
        stock = c5.slider("Stock Option Level (0 to 3)", 0, 3, int(val("StockOptionLevel", 0)))

        ot_opts = ["Yes", "No"]
        overtime = c6.selectbox("Works Overtime?", ot_opts, index=ot_opts.index(val("OverTime", "Yes")))
        travel_opts = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
        travel = c6.selectbox("Business Travel Frequency", travel_opts, index=travel_opts.index(val("BusinessTravel", "Travel_Rarely")))
        perf = c6.slider("Performance Rating (3=Excellent, 4=Outstanding)", 3, 4, int(val("PerformanceRating", 3)))

        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin:20px 0 12px;">'
            '<span class="tag">Section 3</span><span style="font-weight:800;font-size:15px;color:#0F172A;">Tenure, Satisfaction & Workplace Engagement</span></div>',
            unsafe_allow_html=True,
        )
        c7, c8, c9 = st.columns(3)
        years_co = c7.number_input("Years at Company", 0, 40, int(val("YearsAtCompany", 4)))
        years_role = c7.number_input("Years in Current Role", 0, 20, int(val("YearsInCurrentRole", 2)))
        total_work = c7.number_input("Total Working Years", 0, 45, int(val("TotalWorkingYears", 8)))

        last_promo = c8.number_input("Years Since Last Promotion", 0, 20, int(val("YearsSinceLastPromotion", 2)))
        years_mgr = c8.number_input("Years With Current Manager", 0, 20, int(val("YearsWithCurrManager", 2)))
        num_cos = c8.number_input("Number of Companies Worked", 0, 10, int(val("NumCompaniesWorked", 8)))

        sat = c9.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, int(val("JobSatisfaction", 2)))
        env_sat = c9.slider("Environment Satisfaction (1=Low, 4=High)", 1, 4, int(val("EnvironmentSatisfaction", 2)))
        wlb = c9.slider("Work-Life Balance (1=Bad, 4=Best)", 1, 4, int(val("WorkLifeBalance", 1)))
        involvement = c9.slider("Job Involvement (1=Low, 4=High)", 1, 4, int(val("JobInvolvement", 3)))
        rel_sat = c9.slider("Relationship Satisfaction (1=Low, 4=High)", 1, 4, int(val("RelationshipSatisfaction", 1)))

        submitted = st.form_submit_button("⚡ Predict Attrition Risk", use_container_width=True)

    current_sample = {
        "Age": age, "BusinessTravel": travel, "DailyRate": int(val("DailyRate", 800)), "Department": dept,
        "DistanceFromHome": distance, "Education": education, "EducationField": edu_field,
        "EnvironmentSatisfaction": env_sat, "Gender": gender, "HourlyRate": int(val("HourlyRate", 65)),
        "JobInvolvement": involvement, "JobLevel": job_level, "JobRole": role,
        "JobSatisfaction": sat, "MaritalStatus": marital, "MonthlyIncome": income,
        "MonthlyRate": int(val("MonthlyRate", 15000)), "NumCompaniesWorked": num_cos, "OverTime": overtime,
        "PercentSalaryHike": hike, "PerformanceRating": perf,
        "RelationshipSatisfaction": rel_sat, "StockOptionLevel": stock,
        "TotalWorkingYears": total_work, "TrainingTimesLastYear": int(val("TrainingTimesLastYear", 2)),
        "WorkLifeBalance": wlb, "YearsAtCompany": years_co,
        "YearsInCurrentRole": years_role, "YearsSinceLastPromotion": last_promo,
        "YearsWithCurrManager": years_mgr,
    }

    # Automatically compute prediction on load or on submit
    with st.spinner("Executing calibrated SVM inference..."):
        prob = predict_probability(b, current_sample)
        st.session_state["last_sample"] = current_sample
        st.session_state["last_prob"] = prob

    is_leaver = prob >= threshold
    tier = risk_level(prob)
    c_color = RISK[tier]

    st.markdown('<div class="section-label">Prediction Results & Risk Diagnosis</div>', unsafe_allow_html=True)

    # Big Verdict Banner
    if is_leaver:
        st.markdown(
            f"""
            <div class="verdict-box high">
              <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
                <div>
                  <div style="font-size:20px;font-weight:900;color:#991B1B;">
                    🚨 PREDICTION: HIGH ATTRITION RISK (LIKELY TO LEAVE)
                  </div>
                  <div style="font-size:13.5px;color:#7F1D1D;margin-top:4px;">
                    This employee's predicted attrition probability of <b>{prob:.1%}</b> exceeds the optimal decision threshold (<b>{threshold:.1%}</b>).
                    Immediate HR retention intervention is strongly recommended.
                  </div>
                </div>
                <div style="text-align:right;">
                  <span class="pill" style="background:#EF4444;color:#fff;">{tier} Risk Tier</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="verdict-box low">
              <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
                <div>
                  <div style="font-size:20px;font-weight:900;color:#166534;">
                    ✅ PREDICTION: LOW ATTRITION RISK (LIKELY TO STAY)
                  </div>
                  <div style="font-size:13.5px;color:#14532D;margin-top:4px;">
                    This employee's predicted attrition probability of <b>{prob:.1%}</b> is safely below the decision threshold (<b>{threshold:.1%}</b>).
                    Standard engagement and retention practices are sufficient.
                  </div>
                </div>
                <div style="text-align:right;">
                  <span class="pill" style="background:#10B981;color:#fff;">{tier} Risk Tier</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4 Key Metric Cards
    k_cols = st.columns(4)
    k_cols[0].markdown(kpi_card("Attrition Probability", f"{prob:.1%}", f"Threshold: {threshold:.1%}", "📈", c_color, f"{c_color}1A"), unsafe_allow_html=True)
    k_cols[1].markdown(kpi_card("Risk Classification", tier, "Calibrated Tiering", "🎯", c_color, f"{c_color}1A"), unsafe_allow_html=True)
    margin = (prob - threshold) * 100
    margin_str = f"+{margin:.1f}% Above" if margin >= 0 else f"{abs(margin):.1f}% Below"
    k_cols[2].markdown(kpi_card("Decision Margin", margin_str, "vs Decision Threshold", "⚖️", PRIMARY, "#EFF6FF"), unsafe_allow_html=True)
    rec_text = "Intervention Urgent" if is_leaver else "Standard Engagement"
    k_cols[3].markdown(kpi_card("Recommended Action", rec_text, "HR Protocol", "🛡️", VIOLET, "#F5F3FF"), unsafe_allow_html=True)

    # Probability Meter
    st.markdown(
        f"""
        <div class="panel" style="margin-top:16px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="font-size:14px;font-weight:800;color:#0F172A;">Attrition Risk Probability Meter</div>
            <div style="font-size:18px;font-weight:900;color:{c_color};">{prob:.1%}</div>
          </div>
          {seg_gauge(prob, c_color)}
          <div style="display:flex;justify-content:space-between;font-size:11.5px;color:#64748B;margin-top:8px;">
            <span>0% (Strong Retention)</span>
            <span style="color:#2563EB;font-weight:700;">▲ Optimal Threshold ({threshold:.1%})</span>
            <span>100% (Certain Churn)</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # SHAP Explainability & Recommendations Row
    exp_col, rec_col = st.columns([1.2, 1])

    with exp_col:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">Explainability: Top Drivers of This Prediction</div>
              <div class="panel-sub">Sub-second SHAP kernel attribution showing what pushes this employee toward leaving or staying.</div>
            """,
            unsafe_allow_html=True,
        )
        try:
            with st.spinner("Decomposing SHAP contributions..."):
                factors, _ = explain_prediction(b, current_sample, top_n=6, n_samples=100)
            names = [f[0] for f in factors]
            vals = [f[1] for f in factors]
            mx = max(abs(v) for v in vals) if vals else 1.0

            fig_shap = go.Figure(
                go.Bar(
                    y=names,
                    x=vals,
                    orientation="h",
                    marker=dict(
                        color=[RED if v >= 0 else EMERALD for v in vals],
                        cornerradius=5,
                    ),
                    text=[f"{v:+.3f}" for v in vals],
                    textposition="outside",
                    textfont=dict(size=11, color=INK),
                    hovertemplate="%{y}: SHAP %{x:+.3f}<extra></extra>",
                )
            )
            fig_shap.update_xaxes(range=[-mx * 1.35, mx * 1.35], title="SHAP Attribution (Red = Increases Churn, Green = Increases Retention)")
            fig_shap.update_layout(height=310)
            st.plotly_chart(style_fig(fig_shap), use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"SHAP explanation note: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with rec_col:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">Prescriptive Retention Action Plan</div>
              <div class="panel-sub">Automated HR policy recommendations tailored to this employee's risk profile:</div>
              <div style="margin-top:14px;display:flex;flex-direction:column;gap:12px;">
            """,
            unsafe_allow_html=True,
        )

        recs = []
        if overtime == "Yes":
            recs.append("<b>Workload & Overtime:</b> Employee is clocking heavy overtime (+2.3x attrition multiplier). Institute an overtime cap and redistribute project milestones.")
        if sat <= 2:
            recs.append("<b>Stay Interview:</b> Job satisfaction is critically low. Schedule a confidential 1-on-1 retention interview within 7 days to address role friction.")
        if income < 4500:
            recs.append("<b>Compensation Adjustment:</b> Monthly income is below the role median. Review market compensation benchmarks and consider an off-cycle adjustment.")
        if wlb <= 2:
            recs.append("<b>Work-Life Balance:</b> Employee reports strained work-life balance. Offer flexible working hours or partial remote/hybrid scheduling.")
        if distance > 12:
            recs.append("<b>Commute Relief:</b> High commute distance (>12 miles). Provide commute stipend or remote work days to reduce burnout.")
        if last_promo >= 3:
            recs.append("<b>Career Pathing:</b> Over 3 years without promotion. Formulate a transparent 6-month career advancement pathway.")
        if not recs:
            recs.append("<b>Retention Maintenance:</b> Employee exhibits healthy engagement and compensation alignment. Maintain regular quarterly check-ins.")

        for r in recs:
            st.markdown(f'<div style="font-size:13px;line-height:1.6;color:#1E293B;padding-left:10px;border-left:3px solid #2563EB;">{r}</div>', unsafe_allow_html=True)

        st.markdown(
            """
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# PAGE 2: BATCH WORKFORCE CSV PREDICTOR
# ===========================================================================
def render_batch_page():
    b = get_bundle()
    threshold = float(b["threshold"])

    st.markdown(
        """
        <div style="margin-bottom:18px;">
          <div style="font-size:30px;font-weight:900;color:#0F172A;letter-spacing:-.02em;">
            Workforce Batch Attrition Scoring
          </div>
          <div style="font-size:14.5px;color:#64748B;margin-top:4px;">
            Upload an entire company or department CSV roster to predict attrition risk across all employees simultaneously.
            Quantify financial exposure, stratify risk tiers, and export scored rosters.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    up_col1, up_col2 = st.columns([1.6, 1])
    with up_col1:
        uploaded_file = st.file_uploader(
            "Upload Employee Roster (CSV)",
            type=["csv"],
            help="Upload a CSV with employee attributes. The pipeline will automatically map and impute missing fields.",
        )
    with up_col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        use_sample = st.button("⚡ Load 100 Sample Employees (Demo)", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_df):,} employee records from file.")
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")
    elif use_sample:
        raw_full = get_data()
        batch_df = raw_full.sample(n=min(100, len(raw_full)), random_state=42).reset_index(drop=True)
        st.session_state["sample_batch_df"] = batch_df

    if batch_df is None and "sample_batch_df" in st.session_state:
        batch_df = st.session_state["sample_batch_df"]

    if batch_df is not None:
        with st.spinner("Executing batch predictions across workforce..."):
            scored_df = batch_predict(b, batch_df)

        total_emp = len(scored_df)
        n_churn = (scored_df["Attrition_Predicted"] == "Yes").sum()
        churn_pct = n_churn / total_emp if total_emp else 0.0
        n_crit_high = scored_df["Risk_Level"].isin(["High", "Critical"]).sum()

        # Estimate turnover cost ($): Average replacement cost is ~1.5x annual salary (18x monthly income)
        if "MonthlyIncome" in scored_df.columns:
            leavers = scored_df[scored_df["Attrition_Predicted"] == "Yes"]
            est_cost = (leavers["MonthlyIncome"] * 18 * 0.75).sum()
        else:
            est_cost = n_churn * 45000

        st.markdown('<div class="section-label">Workforce Risk Summary</div>', unsafe_allow_html=True)
        b_c = st.columns(4)
        b_c[0].markdown(kpi_card("Total Assessed", f"{total_emp:,}", "Employees in Roster", "👥", PRIMARY, "#EFF6FF"), unsafe_allow_html=True)
        b_c[1].markdown(kpi_card("Predicted Leavers", f"{n_churn}", f"{churn_pct:.1%} of Headcount", "⚠️", RED, "#FEF2F2"), unsafe_allow_html=True)
        b_c[2].markdown(kpi_card("High/Critical Risk", f"{n_crit_high}", "Require Urgent Action", "🔥", AMBER, "#FEF3C7"), unsafe_allow_html=True)
        b_c[3].markdown(kpi_card("Financial Exposure", f"${est_cost/1000:,.0f}K", "Turnover Cost Impact", "💰", VIOLET, "#F5F3FF"), unsafe_allow_html=True)

        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            st.markdown('<div class="panel-title" style="margin:16px 0 8px;">Workforce Risk Tier Breakdown</div>', unsafe_allow_html=True)
            risk_counts = scored_df["Risk_Level"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0)
            fig_risk = go.Figure(
                go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    hole=0.6,
                    marker=dict(colors=[EMERALD, AMBER, RED, "#7F1D1D"]),
                    textinfo="label+value",
                    textfont=dict(size=12),
                )
            )
            st.plotly_chart(style_fig(fig_risk, 260), use_container_width=True, config={"displayModeBar": False})

        with ch_col2:
            st.markdown('<div class="panel-title" style="margin:16px 0 8px;">Predicted Risk by Department</div>', unsafe_allow_html=True)
            if "Department" in scored_df.columns:
                dep_risk = scored_df.groupby("Department")["Attrition_Probability"].mean() * 100
                fig_dep = go.Figure(
                    go.Bar(
                        x=dep_risk.values,
                        y=dep_risk.index,
                        orientation="h",
                        marker=dict(color=PRIMARY, cornerradius=5),
                        text=[f"{v:.1f}%" for v in dep_risk.values],
                        textposition="outside",
                    )
                )
                fig_dep.update_xaxes(title="Avg Attrition Probability (%)")
                st.plotly_chart(style_fig(fig_dep, 260), use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-label">Ranked At-Risk Employees</div>', unsafe_allow_html=True)
        display_cols = [
            c for c in [
                "EmployeeNumber", "JobRole", "Department", "MonthlyIncome",
                "OverTime", "WorkLifeBalance", "Attrition_Probability",
                "Risk_Level", "Attrition_Predicted"
            ] if c in scored_df.columns
        ]
        ranked_df = scored_df.sort_values("Attrition_Probability", ascending=False)
        st.dataframe(ranked_df[display_cols].head(50), width="stretch", hide_index=True)

        csv_buf = io.StringIO()
        scored_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="📥 Download Scored Workforce Roster (CSV)",
            data=csv_buf.getvalue(),
            file_name="employee_attrition_scored.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# PAGE 3: WHAT-IF RETENTION INTERVENTION SIMULATOR
# ===========================================================================
def render_simulator_page():
    b = get_bundle()
    threshold = float(b["threshold"])

    st.markdown(
        """
        <div style="margin-bottom:18px;">
          <div style="font-size:30px;font-weight:900;color:#0F172A;letter-spacing:-.02em;">
            Interactive Retention Intervention Simulator
          </div>
          <div style="font-size:14.5px;color:#64748B;margin-top:4px;">
            Simulate proactive HR retention policies on at-risk employees. Quantify the probability reduction
            and return on investment (ROI) before executing organizational changes.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Base profile selection
    sim_preset_key = st.selectbox(
        "Select Baseline Employee Profile to Simulate Interventions:",
        [k for k in PRESETS.keys() if PRESETS[k] is not None],
        index=0,
    )
    base_sample = dict(PRESETS[sim_preset_key])

    base_prob = predict_probability(b, base_sample)
    base_tier = risk_level(base_prob)

    st.markdown('<div class="section-label">Simulate HR Interventions</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        st.markdown("<b>Work Hours & Flexibility:</b>", unsafe_allow_html=True)
        eliminate_ot = st.checkbox(
            "Eliminate Mandatory Overtime (OverTime: Yes ➔ No)",
            value=(base_sample.get("OverTime") == "Yes"),
        )
        improve_wlb = st.slider(
            "Improve Work-Life Balance Rating",
            1, 4,
            max(base_sample.get("WorkLifeBalance", 1), 3),
        )
        hybrid_commute = st.checkbox(
            "Offer Hybrid Remote Schedule (Effective Commute ➔ 2 Miles)",
            value=False,
        )

    with sim_col2:
        st.markdown("<b>Compensation & Career Progression:</b>", unsafe_allow_html=True)
        salary_boost_pct = st.slider(
            "Simulate Salary Revision (%)",
            0, 35, 15, step=5,
        )
        fast_track_promo = st.checkbox(
            "Career Advancement Review (Reset Years Since Last Promotion ➔ 0)",
            value=False,
        )
        grant_stock = st.checkbox(
            "Grant Retention Stock Options (Stock Option Level ➔ 2)",
            value=False,
        )

    # Apply simulated interventions to sample copy
    sim_sample = dict(base_sample)
    if eliminate_ot:
        sim_sample["OverTime"] = "No"
    sim_sample["WorkLifeBalance"] = improve_wlb
    if hybrid_commute:
        sim_sample["DistanceFromHome"] = 2
    if salary_boost_pct > 0:
        sim_sample["MonthlyIncome"] = int(base_sample["MonthlyIncome"] * (1 + salary_boost_pct / 100.0))
        sim_sample["PercentSalaryHike"] = min(25, base_sample["PercentSalaryHike"] + 5)
    if fast_track_promo:
        sim_sample["YearsSinceLastPromotion"] = 0
    if grant_stock:
        sim_sample["StockOptionLevel"] = max(base_sample.get("StockOptionLevel", 0), 2)

    sim_prob = predict_probability(b, sim_sample)
    sim_tier = risk_level(sim_prob)
    prob_drop = (base_prob - sim_prob) * 100

    st.markdown("</div>", unsafe_allow_html=True)

    # Comparison Cards
    st.markdown('<div class="section-label">Intervention Impact Analysis</div>', unsafe_allow_html=True)
    res_c1, res_c2, res_c3 = st.columns(3)

    res_c1.markdown(
        f"""
        <div class="kpi">
          <div class="accent" style="background:{RISK[base_tier]};"></div>
          <div class="lbl">Baseline Attrition Risk</div>
          <div class="val" style="color:{RISK[base_tier]};">{base_prob:.1%}</div>
          <div class="sub">Current Risk Tier: <b>{base_tier}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    res_c2.markdown(
        f"""
        <div class="kpi">
          <div class="accent" style="background:{RISK[sim_tier]};"></div>
          <div class="lbl">Post-Intervention Risk</div>
          <div class="val" style="color:{RISK[sim_tier]};">{sim_prob:.1%}</div>
          <div class="sub">Simulated Tier: <b>{sim_tier}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    delta_color = EMERALD if prob_drop > 0 else RED
    res_c3.markdown(
        f"""
        <div class="kpi">
          <div class="accent" style="background:{delta_color};"></div>
          <div class="lbl">Risk Reduction Delta</div>
          <div class="val" style="color:{delta_color};">-{prob_drop:.1f}%</div>
          <div class="sub">{'✅ Moved below threshold into Safe zone' if sim_prob < threshold and base_prob >= threshold else 'Risk significantly mitigated'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Visual Comparison Bar
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=["Before Interventions", "After Interventions"],
        y=[base_prob * 100, sim_prob * 100],
        marker_color=[RISK[base_tier], RISK[sim_tier]],
        text=[f"{base_prob:.1%}", f"{sim_prob:.1%}"],
        textposition="outside",
        textfont=dict(size=14, weight=900),
        width=0.45,
    ))
    fig_comp.add_hline(
        y=threshold * 100,
        line_dash="dash",
        line_color="#2563EB",
        annotation_text=f"Decision Threshold ({threshold:.1%})",
        annotation_position="top left",
    )
    fig_comp.update_yaxes(title="Attrition Probability (%)", range=[0, 105])
    fig_comp.update_layout(height=320)
    st.plotly_chart(style_fig(fig_comp), use_container_width=True, config={"displayModeBar": False})

    # Financial ROI Summary
    salary = base_sample.get("MonthlyIncome", 5000) * 12
    est_turnover_cost = salary * 1.5
    est_intervention_cost = (salary * (salary_boost_pct / 100.0)) + 3000
    net_savings = est_turnover_cost - est_intervention_cost

    st.markdown(
        f"""
        <div class="panel" style="background:#F0FDF4;border:1px solid #BBF7D0;">
          <div style="font-size:16px;font-weight:900;color:#166534;">💼 Business Case & Cost Savings Estimate</div>
          <div style="font-size:13.5px;color:#14532D;margin-top:6px;line-height:1.7;">
            By reducing attrition probability from <b>{base_prob:.1%}</b> to <b>{sim_prob:.1%}</b>, the organization avoids an estimated
            <b>${est_turnover_cost:,.0f}</b> turnover replacement expense for an intervention investment of approximately <b>${est_intervention_cost:,.0f}</b>.
            <br><b>Estimated Net Retention Value: +${net_savings:,.0f}</b>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 4: WORKFORCE HISTORICAL DATA & EDA
# ===========================================================================
def render_eda_page():
    df = get_data()
    b = get_bundle()
    rate = df["Attrition"].eq("Yes").mean()

    st.markdown(
        f"""
        <div style="margin-bottom:18px;">
          <div style="font-size:30px;font-weight:900;color:#0F172A;letter-spacing:-.02em;">
            Workforce Historical Analytics & Exploratory Data Analysis
          </div>
          <div style="font-size:14.5px;color:#64748B;margin-top:4px;">
            Empirical baseline analysis on the 1,470 employee records from the IBM HR Analytics benchmark dataset.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c = st.columns(4)
    c[0].markdown(kpi_card("Total Workforce", f"{len(df):,}", "Historical records", "👥", PRIMARY, "#EFF6FF"), unsafe_allow_html=True)
    c[1].markdown(kpi_card("Baseline Attrition", f"{rate:.1%}", "237 total leavers", "📉", RED, "#FEF2F2"), unsafe_allow_html=True)
    c[2].markdown(kpi_card("Overtime Prevalence", f"{df['OverTime'].eq('Yes').mean():.1%}", "416 employees clock OT", "⏰", AMBER, "#FEF3C7"), unsafe_allow_html=True)
    c[3].markdown(kpi_card("Avg Monthly Income", f"${df['MonthlyIncome'].mean():,.0f}", "Across all roles", "💵", EMERALD, "#ECFDF5"), unsafe_allow_html=True)

    st.markdown('<div class="section-label">Empirical Drivers</div>', unsafe_allow_html=True)
    r1 = st.columns(2)
    with r1[0]:
        st.markdown('<div class="panel"><div class="panel-title">Attrition Distribution (Class Imbalance)</div>', unsafe_allow_html=True)
        counts = df["Attrition"].value_counts()
        fig = go.Figure(
            go.Pie(
                labels=["Stayed (Retained)", "Left (Attrition)"],
                values=counts.values,
                hole=0.6,
                marker=dict(colors=[PRIMARY, RED]),
                textinfo="label+percent",
            )
        )
        st.plotly_chart(style_fig(fig, 280), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with r1[1]:
        st.markdown('<div class="panel"><div class="panel-title">Attrition Rate by Overtime Status</div>', unsafe_allow_html=True)
        ot = df.groupby("OverTime")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)
        fig = go.Figure(
            go.Bar(
                x=ot.values,
                y=["No Overtime", "Works Overtime"],
                orientation="h",
                marker=dict(color=[PRIMARY, RED], cornerradius=6),
                text=[f"{v:.1f}%" for v in ot.values],
                textposition="outside",
            )
        )
        fig.update_xaxes(title="Attrition Rate (%)")
        st.plotly_chart(style_fig(fig, 280), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    r2 = st.columns(2)
    with r2[0]:
        st.markdown('<div class="panel"><div class="panel-title">Attrition by Job Role</div>', unsafe_allow_html=True)
        jr = df.groupby("JobRole")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100).sort_values()
        fig = go.Figure(
            go.Bar(
                x=jr.values,
                y=jr.index,
                orientation="h",
                marker=dict(color=PRIMARY, cornerradius=5),
                text=[f"{v:.1f}%" for v in jr.values],
                textposition="outside",
            )
        )
        fig.update_xaxes(title="Attrition Rate (%)")
        st.plotly_chart(style_fig(fig, 320), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with r2[1]:
        st.markdown('<div class="panel"><div class="panel-title">Attrition by Work-Life Balance Rating</div>', unsafe_allow_html=True)
        wlb = df.groupby("WorkLifeBalance")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)
        fig = go.Figure(
            go.Scatter(
                x=wlb.index,
                y=wlb.values,
                mode="lines+markers+text",
                line=dict(color=AMBER, width=3),
                marker=dict(size=9, color=AMBER),
                text=[f"{v:.1f}%" for v in wlb.values],
                textposition="top center",
            )
        )
        fig.update_layout(xaxis=dict(tickmode="array", tickvals=[1, 2, 3, 4], title="Rating (1=Bad, 4=Best)"), yaxis_title="Attrition Rate (%)")
        st.plotly_chart(style_fig(fig, 320), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# PAGE 5: MODEL PERFORMANCE & BENCHMARK
# ===========================================================================
def render_models_page():
    r = get_results()
    fm = r["final_metrics"]

    st.markdown(
        """
        <div style="margin-bottom:18px;">
          <div style="font-size:30px;font-weight:900;color:#0F172A;letter-spacing:-.02em;">
            Machine Learning Model Evaluation & Benchmark
          </div>
          <div style="font-size:14.5px;color:#64748B;margin-top:4px;">
            Rigorous evaluation of five supervised learning algorithms across 441 test instances (stratified 30% hold-out).
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c = st.columns(6)
    c[0].markdown(kpi_card("Champion Model", "SVM (RBF)", "Tuned Hyperparameters", "🏆", VIOLET, "#F5F3FF"), unsafe_allow_html=True)
    c[1].markdown(kpi_card("Test Accuracy", f"{fm['accuracy']:.2%}", "Overall accuracy", "🎯", PRIMARY, "#EFF6FF"), unsafe_allow_html=True)
    c[2].markdown(kpi_card("Test Precision", f"{fm['precision']:.2%}", "Minimizes false alarms", "🎯", PRIMARY, "#EFF6FF"), unsafe_allow_html=True)
    c[3].markdown(kpi_card("Test Recall", f"{fm['recall']:.2%}", "Catches true leavers", "🎯", PRIMARY, "#EFF6FF"), unsafe_allow_html=True)
    c[4].markdown(kpi_card("Balanced F1", f"{fm['f1']:.2%}", "Primary selection metric", "⭐", AMBER, "#FEF3C7"), unsafe_allow_html=True)
    c[5].markdown(kpi_card("ROC-AUC", f"{fm['roc_auc']:.3f}", "Discrimination power", "🧭", EMERALD, "#ECFDF5"), unsafe_allow_html=True)

    st.markdown('<div class="section-label">Model Benchmark Comparison</div>', unsafe_allow_html=True)
    rows = []
    for name, m in r["all_models"].items():
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "ROC-AUC": m["roc_auc"],
            "CV F1": m["cv_f1_mean"],
        })
    table = pd.DataFrame(rows).sort_values("F1", ascending=False)

    fig_f1 = go.Figure(
        go.Bar(
            y=[short_model(n) for n in table["Model"]],
            x=table["F1"],
            orientation="h",
            marker=dict(color=[PRIMARY if n == r["final_model"] else "#CBD5E1" for n in table["Model"]], cornerradius=5),
            text=[f"{v:.3f}" for v in table["F1"]],
            textposition="outside",
        )
    )
    fig_f1.update_xaxes(title="F1 Score (Balanced)", range=[0, table["F1"].max() * 1.2])
    st.plotly_chart(style_fig(fig_f1, 280), use_container_width=True, config={"displayModeBar": False})

    tab_cm, tab_roc, tab_tbl = st.tabs(["Confusion Matrix (Standard)", "ROC Curves", "Full Metrics Table"])
    with tab_cm:
        cm = r["confusion_data"][r["final_model"]]
        z = np.array(cm)
        fig_cm = go.Figure(
            go.Heatmap(
                z=z,
                x=["Predicted Stay", "Predicted Leave"],
                y=["Actual Stay", "Actual Leave"],
                colorscale="Blues",
                showscale=False,
                text=[[f"{int(v):,}" for v in row] for row in z],
                texttemplate="%{text}",
                textfont=dict(size=18, color="#fff"),
            )
        )
        fig_cm.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Predicted Outcome", yaxis_title="Actual Ground Truth")
        st.plotly_chart(style_fig(fig_cm, 320), use_container_width=True, config={"displayModeBar": False})

    with tab_roc:
        fig_roc = go.Figure()
        for name, d in r["roc_data"].items():
            fig_roc.add_trace(go.Scatter(
                x=d["fpr"], y=d["tpr"], mode="lines",
                name=f"{short_model(name)} (AUC {r['all_models'][name]['roc_auc']:.3f})",
                line=dict(width=2.5 if name == r["final_model"] else 1.5),
            ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(color="#CBD5E1", dash="dash")))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(style_fig(fig_roc, 360), use_container_width=True, config={"displayModeBar": False})

    with tab_tbl:
        st.dataframe(table.round(3), width="stretch", hide_index=True)


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================
selected_view = sidebar_navigation()

if selected_view == "🎯 Predict Employee Attrition":
    render_predict_page()
elif selected_view == "📁 Batch Workforce CSV Predictor":
    render_batch_page()
elif selected_view == "🔄 Retention Intervention Simulator":
    render_simulator_page()
elif selected_view == "📊 Workforce Historical Data & EDA":
    render_eda_page()
elif selected_view == "🧠 Model Performance & Validation":
    render_models_page()