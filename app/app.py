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
PRIMARY_DK = "#1D4ED8"
EMERALD = "#10B981"
RED = "#EF4444"
AMBER = "#F59E0B"
VIOLET = "#8B5CF6"

RISK = {"Low": EMERALD, "Medium": AMBER, "High": RED, "Critical": "#991B1B"}

# ---------------------------------------------------------------------------
# HUMAN-FRIENDLY FEATURE METADATA & DISPLAY FORMATTERS
# ---------------------------------------------------------------------------
FEATURE_METADATA = {
    "WorkLifeBalance": {
        "name": "Work-Life Balance",
        "icon": "⚖️",
        "format": lambda v: {1: "Poor (Burnout Risk)", 2: "Below Average", 3: "Good / Balanced", 4: "Excellent"}.get(v, f"Rating {v}"),
        "desc": "Balance between working demands and personal life.",
    },
    "OverTime": {
        "name": "Overtime Status",
        "icon": "⏰",
        "format": lambda v: "Yes (Works Overtime)" if v == "Yes" else "No (Standard 40h)",
        "desc": "Whether employee routinely works beyond standard hours.",
    },
    "MonthlyIncome": {
        "name": "Monthly Salary",
        "icon": "💵",
        "format": lambda v: f"${int(v):,}/mo",
        "desc": "Base gross monthly salary in USD.",
    },
    "JobSatisfaction": {
        "name": "Job Satisfaction",
        "icon": "😊",
        "format": lambda v: {1: "Very Low", 2: "Low / Dissatisfied", 3: "Satisfied", 4: "Very Satisfied"}.get(v, f"Rating {v}"),
        "desc": "Self-reported satisfaction with day-to-day job duties.",
    },
    "EnvironmentSatisfaction": {
        "name": "Workplace Environment",
        "icon": "🏢",
        "format": lambda v: {1: "Low Satisfaction", 2: "Medium", 3: "High", 4: "Very High"}.get(v, f"Rating {v}"),
        "desc": "Satisfaction with physical work culture and team facilities.",
    },
    "DistanceFromHome": {
        "name": "Commute Distance",
        "icon": "🚗",
        "format": lambda v: f"{int(v)} miles",
        "desc": "One-way commute distance from residence to workplace.",
    },
    "StockOptionLevel": {
        "name": "Stock Option Equity",
        "icon": "📈",
        "format": lambda v: {0: "Level 0 (No Equity)", 1: "Level 1", 2: "Level 2", 3: "Level 3 (Max)"}.get(v, f"Level {v}"),
        "desc": "Equity vesting program level granted to employee.",
    },
    "TotalWorkingYears": {
        "name": "Career Experience",
        "icon": "💼",
        "format": lambda v: f"{int(v)} years total",
        "desc": "Total accumulated professional work experience.",
    },
    "YearsAtCompany": {
        "name": "Tenure at Company",
        "icon": "🏛️",
        "format": lambda v: f"{int(v)} years",
        "desc": "Years continuously employed at current organization.",
    },
    "YearsInCurrentRole": {
        "name": "Years in Current Role",
        "icon": "🎯",
        "format": lambda v: f"{int(v)} years",
        "desc": "Years spent in the current job title without lateral shift.",
    },
    "YearsSinceLastPromotion": {
        "name": "Years Since Last Promotion",
        "icon": "🚀",
        "format": lambda v: f"{int(v)} years",
        "desc": "Years since last official upward title or grade promotion.",
    },
    "YearsWithCurrManager": {
        "name": "Years With Current Manager",
        "icon": "🤝",
        "format": lambda v: f"{int(v)} years",
        "desc": "Supervisory continuity and tenure under active manager.",
    },
    "NumCompaniesWorked": {
        "name": "Prior Companies Worked",
        "icon": "🏢",
        "format": lambda v: f"{int(v)} previous employers",
        "desc": "Number of previous companies worked for before joining.",
    },
    "Department": {
        "name": "Department",
        "icon": "🏷️",
        "format": lambda v: str(v),
        "desc": "Functional organizational division.",
    },
    "JobRole": {
        "name": "Job Role / Title",
        "icon": "👔",
        "format": lambda v: str(v),
        "desc": "Specific occupational position.",
    },
    "MaritalStatus": {
        "name": "Marital Status",
        "icon": "💍",
        "format": lambda v: str(v),
        "desc": "Marital status (Single, Married, or Divorced).",
    },
    "BusinessTravel": {
        "name": "Business Travel",
        "icon": "✈️",
        "format": lambda v: {"Travel_Rarely": "Travels Rarely", "Travel_Frequently": "Travels Frequently", "Non-Travel": "No Travel"}.get(v, str(v)),
        "desc": "Frequency of required work-related business travel.",
    },
    "PercentSalaryHike": {
        "name": "Annual Salary Hike",
        "icon": "📊",
        "format": lambda v: f"{int(v)}% increase",
        "desc": "Percentage salary increase awarded during last review.",
    },
    "Age": {
        "name": "Employee Age",
        "icon": "👤",
        "format": lambda v: f"{int(v)} years old",
        "desc": "Chronological age in years.",
    },
    "JobInvolvement": {
        "name": "Job Involvement",
        "icon": "🔥",
        "format": lambda v: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}.get(v, f"Rating {v}"),
        "desc": "Degree of psychological identification with one's work.",
    },
    "RelationshipSatisfaction": {
        "name": "Colleague Relationships",
        "icon": "🤝",
        "format": lambda v: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}.get(v, f"Rating {v}"),
        "desc": "Interpersonal satisfaction with coworkers and stakeholders.",
    },
    "Education": {
        "name": "Education Level",
        "icon": "🎓",
        "format": lambda v: {1: "Below College", 2: "Associate / College", 3: "Bachelor Degree", 4: "Master Degree", 5: "Doctorate"}.get(v, f"Level {v}"),
        "desc": "Highest completed academic credential.",
    },
    "EducationField": {
        "name": "Education Field",
        "icon": "📚",
        "format": lambda v: str(v),
        "desc": "Academic discipline or major.",
    },
    "Gender": {
        "name": "Gender",
        "icon": "⚧️",
        "format": lambda v: str(v),
        "desc": "Self-identified gender.",
    },
    "JobLevel": {
        "name": "Job Seniority Level",
        "icon": "🎖️",
        "format": lambda v: f"Level {int(v)} (Band)",
        "desc": "Corporate hierarchical grade band (1 to 5).",
    },
    "DailyRate": {
        "name": "Daily Compensation Index",
        "icon": "💲",
        "format": lambda v: f"${int(v)}",
        "desc": "Daily compensation reference index.",
    },
    "HourlyRate": {
        "name": "Hourly Pay Index",
        "icon": "⏱️",
        "format": lambda v: f"${int(v)}/hr",
        "desc": "Hourly wage reference index.",
    },
    "MonthlyRate": {
        "name": "Monthly Billing Index",
        "icon": "📋",
        "format": lambda v: f"${int(v)}",
        "desc": "Monthly billing / administrative cost index.",
    },
    "TrainingTimesLastYear": {
        "name": "Training Sessions Attended",
        "icon": "📖",
        "format": lambda v: f"{int(v)} sessions",
        "desc": "Professional training and upskilling sessions completed last year.",
    },
    "PerformanceRating": {
        "name": "Performance Review Rating",
        "icon": "⭐",
        "format": lambda v: {3: "Excellent", 4: "Outstanding"}.get(v, f"Rating {v}"),
        "desc": "Official annual performance appraisal score.",
    },
}

# ---------------------------------------------------------------------------
# REALISTIC EMPLOYEE DIRECTORY & PERSONA PRESETS
# ---------------------------------------------------------------------------
EMPLOYEE_DIRECTORY = {
    "EMP-1042": {
        "id": "EMP-1042",
        "name": "Sarah Jenkins",
        "role": "Sales Executive",
        "department": "Sales",
        "avatar": "SJ",
        "label": "👤 Sarah Jenkins — Sales Executive · Sales Dept",
        "summary": "Mid-level sales executive with 4 years tenure. Experiences regular overtime, high commute distance, and low work-life balance.",
        "data": {
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
    },
    "EMP-1015": {
        "id": "EMP-1015",
        "name": "Alex Rivera",
        "role": "Sales Representative",
        "department": "Sales",
        "avatar": "AR",
        "label": "👤 Alex Rivera — Sales Representative · Sales Dept",
        "summary": "Junior sales representative with entry compensation ($2,400), frequent business travel, 24-mile commute, and mandatory overtime.",
        "data": {
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
    },
    "EMP-1088": {
        "id": "EMP-1088",
        "name": "David Chen",
        "role": "Laboratory Technician",
        "department": "Research & Development",
        "avatar": "DC",
        "label": "👤 David Chen — Laboratory Technician · R&D Dept",
        "summary": "Early-career technician with 3 years tenure. No overtime and balanced commute, but modest salary ($3,200) and stagnant promotion track.",
        "data": {
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
    },
    "EMP-1120": {
        "id": "EMP-1120",
        "name": "Dr. Elena Rostova",
        "role": "Manufacturing Director",
        "department": "Research & Development",
        "avatar": "ER",
        "label": "👤 Dr. Elena Rostova — Manufacturing Director · R&D Dept",
        "summary": "Senior engineering leader with 12 years tenure, top-tier compensation ($12,800), high equity grant, and excellent satisfaction.",
        "data": {
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
    },
    "EMP-1155": {
        "id": "EMP-1155",
        "name": "Marcus Vance",
        "role": "Research Scientist",
        "department": "Research & Development",
        "avatar": "MV",
        "label": "👤 Marcus Vance — Research Scientist · R&D Dept",
        "summary": "Established scientist with 7 years tenure, strong peer relationships, standard 40h work week, and stable compensation ($6,500).",
        "data": {
            "Age": 36, "BusinessTravel": "Travel_Rarely", "DailyRate": 800, "Department": "Research & Development",
            "DistanceFromHome": 7, "Education": 3, "EducationField": "Medical",
            "EnvironmentSatisfaction": 3, "Gender": "Male", "HourlyRate": 66,
            "JobInvolvement": 3, "JobLevel": 2, "JobRole": "Research Scientist",
            "JobSatisfaction": 3, "MaritalStatus": "Married", "MonthlyIncome": 6500,
            "MonthlyRate": 14000, "NumCompaniesWorked": 2, "OverTime": "No",
            "PercentSalaryHike": 15, "PerformanceRating": 3,
            "RelationshipSatisfaction": 3, "StockOptionLevel": 1,
            "TotalWorkingYears": 11, "TrainingTimesLastYear": 3,
            "WorkLifeBalance": 3, "YearsAtCompany": 7,
            "YearsInCurrentRole": 4, "YearsSinceLastPromotion": 2,
            "YearsWithCurrManager": 4,
        },
    },
    "CUSTOM": {
        "id": "NEW-EMP",
        "name": "Custom Employee",
        "role": "Custom Position",
        "department": "Custom Department",
        "avatar": "✍️",
        "label": "✍️ Custom Assessment (Enter Details Manually)",
        "summary": "Manual assessment mode: customize demographic, compensation, and satisfaction attributes below.",
        "data": None,
    },
}

st.set_page_config(
    page_title="AttritionIQ | AI Employee Turnover Intelligence",
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
.block-container { padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1280px; }

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #091322 0%, #0F2038 50%, #162E50 100%);
  border-right: 1px solid rgba(255,255,255,.08);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
div[role="radiogroup"] { gap: 6px; }
div[role="radiogroup"] label {
  padding: 0.70rem 0.90rem !important; border-radius: 12px; cursor: pointer;
  transition: all .18s cubic-bezier(.16,1,.3,1);
}
div[role="radiogroup"] label:hover { background: rgba(255,255,255,.08); transform: translateX(3px); }
div[role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(135deg, rgba(37,99,235,.40), rgba(59,130,246,.25));
  box-shadow: inset 0 0 0 1px rgba(96,165,250,.60), 0 4px 12px rgba(37,99,235,.20);
}
div[role="radiogroup"] p, div[role="radiogroup"] [data-testid="stMarkdownContainer"] p {
  font-weight: 700 !important; font-size: 13.5px !important; color: #F1F5F9 !important;
}

/* ---------- Premium Hero Card ---------- */
.hero-card {
  background: linear-gradient(135deg, #0A192F 0%, #11294D 55%, #183C6C 100%);
  border-radius: 20px; padding: 28px 32px; margin-bottom: 22px; position: relative; overflow: hidden;
  box-shadow: 0 12px 30px -8px rgba(10,25,47,.28), 0 4px 12px -2px rgba(10,25,47,.16);
  border: 1px solid rgba(255,255,255,.12);
}
.hero-card::after {
  content: ""; position: absolute; right: -80px; top: -80px; width: 340px; height: 340px;
  border-radius: 50%; background: radial-gradient(circle, rgba(59,130,246,.28), transparent 70%);
  pointer-events: none;
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 8px; background: rgba(255,255,255,.12);
  border: 1px solid rgba(255,255,255,.20); border-radius: 999px; padding: 4px 14px;
  font-size: 11px; font-weight: 800; color: #BFDBFE; text-transform: uppercase; letter-spacing: .08em;
  backdrop-filter: blur(8px); margin-bottom: 12px;
}
.hero-title { font-size: 32px; font-weight: 900; color: #FFFFFF; letter-spacing: -.02em; }
.hero-sub { font-size: 14.5px; color: #CBD5E1; margin-top: 6px; max-width: 780px; line-height: 1.6; }
.hero-stat-row { display: flex; gap: 12px; margin-top: 20px; flex-wrap: wrap; }
.hero-stat-pill {
  display: inline-flex; align-items: baseline; gap: 8px; background: rgba(255,255,255,.08);
  border: 1px solid rgba(255,255,255,.14); border-radius: 12px; padding: 8px 16px; backdrop-filter: blur(6px);
}
.hero-stat-pill .num { font-size: 18px; font-weight: 900; color: #FFFFFF; }
.hero-stat-pill .lbl { font-size: 11px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: #93C5FD; }

/* ---------- Form Inputs & High Contrast Labels ---------- */
label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] * {
  color: #0F172A !important;
  font-weight: 800 !important;
  font-size: 13.5px !important;
  opacity: 1 !important;
  visibility: visible !important;
  display: block !important;
  margin-bottom: 5px !important;
}

[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input {
  background-color: #FFFFFF !important;
  color: #0F172A !important;
  border: 1.5px solid #CBD5E1 !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  box-shadow: 0 1px 3px rgba(0,0,0,.03) !important;
}

[data-testid="stNumberInput"] input:focus,
div[data-baseweb="select"] > div:focus-within {
  border-color: #2563EB !important;
  box-shadow: 0 0 0 3px rgba(37,99,235,.15) !important;
}

[data-testid="stNumberInput"] button {
  background-color: #F1F5F9 !important;
  color: #0F172A !important;
  border: 1px solid #CBD5E1 !important;
}

/* ---------- Tabs Styling ---------- */
.stTabs [data-baseweb="tab-list"] {
  gap: 10px; border-bottom: 2px solid #E2E8F0; padding-bottom: 2px;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 12px 12px 0 0; padding: 10px 22px; font-weight: 700; font-size: 14px;
  background: #EEF2F6; color: #475569; transition: all .18s ease;
}
.stTabs [aria-selected="true"] {
  background: #2563EB !important; color: #FFFFFF !important;
  box-shadow: 0 4px 14px rgba(37,99,235,.25);
}

/* ---------- Buttons ---------- */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius: 12px; font-weight: 800; border: none; font-size: 15px;
  background: linear-gradient(135deg,#2563EB 0%, #1D4ED8 100%); color: #fff;
  padding: 0.78em 1.8em; transition: all .2s cubic-bezier(.16,1,.3,1);
  box-shadow: 0 6px 18px rgba(37,99,235,.32);
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
  transform: translateY(-2px); box-shadow: 0 10px 26px rgba(37,99,235,.42); color: #fff;
}

/* ---------- Cards & Panels ---------- */
.panel {
  background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 18px;
  padding: 24px 26px; box-shadow: 0 6px 22px rgba(15,23,42,.04), 0 2px 6px rgba(15,23,42,.02);
  margin-bottom: 22px;
}
.panel-title { font-size: 17px; font-weight: 900; color: #0F172A; display: flex; align-items: center; gap: 8px; }
.panel-sub { font-size: 13px; color: #64748B; margin-top: 4px; line-height: 1.5; }

.kpi {
  background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 18px; padding: 20px 22px;
  box-shadow: 0 6px 20px rgba(15,23,42,.04); position: relative; overflow: hidden;
  transition: all .2s ease;
}
.kpi:hover { transform: translateY(-2px); box-shadow: 0 10px 28px rgba(15,23,42,.07); }
.kpi .accent { position: absolute; left: 0; top: 0; bottom: 0; width: 6px; }
.kpi .icon {
  width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
  font-size: 20px; margin-bottom: 12px;
}
.kpi .lbl { font-size: 11.5px; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; color: #64748B; }
.kpi .val { font-size: 28px; font-weight: 900; color: #0F172A; margin-top: 2px; letter-spacing: -.02em; }
.kpi .sub { font-size: 12.5px; color: #94A3B8; margin-top: 4px; font-weight: 500; }

/* ---------- Employee Profile Card ---------- */
.emp-card {
  background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
  border: 1.5px solid #CBD5E1; border-radius: 18px;
  padding: 22px 26px; margin-bottom: 22px; box-shadow: 0 6px 22px rgba(15,23,42,.05);
  display: flex; gap: 22px; align-items: center;
}
.emp-avatar {
  width: 64px; height: 64px; border-radius: 16px;
  background: linear-gradient(135deg, #2563EB, #7C3AED);
  display: flex; align-items: center; justify-content: center;
  font-size: 24px; font-weight: 900; color: #FFFFFF; flex-shrink: 0;
  box-shadow: 0 6px 18px rgba(37,99,235,.30);
}
.emp-info { flex: 1; }
.emp-name { font-size: 21px; font-weight: 900; color: #0F172A; display: flex; align-items: center; gap: 10px; }
.emp-id-badge {
  background: #EFF6FF; color: #1D4ED8; font-size: 12px; font-weight: 800;
  padding: 4px 12px; border-radius: 8px; border: 1px solid #BFDBFE; letter-spacing: .04em;
}
.emp-role { font-size: 14px; font-weight: 700; color: #475569; margin-top: 4px; }
.emp-summary { font-size: 13px; color: #64748B; margin-top: 8px; line-height: 1.6; }
.emp-chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.emp-chip {
  background: #F1F5F9; border: 1px solid #E2E8F0; border-radius: 8px;
  padding: 4px 12px; font-size: 12px; font-weight: 700; color: #334155;
}

/* ---------- Verdict Banner ---------- */
.verdict-box {
  border-radius: 20px; padding: 26px 30px; margin-bottom: 22px;
  border-width: 2px; border-style: solid;
  box-shadow: 0 8px 26px rgba(15,23,42,.06);
}
.verdict-box.high {
  background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
  border-color: #F87171;
}
.verdict-box.low {
  background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
  border-color: #4ADE80;
}

.pill {
  display: inline-flex; align-items: center; gap: 6px; padding: 7px 18px; border-radius: 999px;
  font-size: 13.5px; font-weight: 900; text-transform: uppercase; letter-spacing: .04em;
  box-shadow: 0 4px 12px rgba(0,0,0,.08);
}
.tag {
  display: inline-block; padding: 4px 12px; border-radius: 8px; font-size: 12px; font-weight: 800;
  letter-spacing: .05em; text-transform: uppercase; background: #E8EFFA; color: #1E40AF;
}
.section-label { font-size: 13px; font-weight: 900; letter-spacing: .08em; text-transform: uppercase; color: #475569; margin: 24px 0 14px; }

/* ---------- Recommendation Cards ---------- */
.rec-card {
  background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px;
  padding: 14px 18px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(15,23,42,.03);
  display: flex; gap: 14px; align-items: flex-start;
}
.rec-badge {
  font-size: 11px; font-weight: 900; text-transform: uppercase; padding: 3px 8px; border-radius: 6px;
  letter-spacing: .04em; flex-shrink: 0; margin-top: 2px;
}
.rec-badge.p1 { background: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }
.rec-badge.p2 { background: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
.rec-badge.p3 { background: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE; }
.rec-text { font-size: 13px; color: #1E293B; line-height: 1.5; font-weight: 500; }
.rec-text b { font-weight: 800; color: #0F172A; }

.mono { font-family: 'JetBrains Mono', monospace; }
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


def render_gauge_chart(probability: float, threshold: float, color: str):
    """Render a modern interactive Plotly half-donut arc gauge."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={
                "suffix": "%",
                "font": {"size": 42, "family": "Inter, sans-serif", "color": INK},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#94A3B8",
                    "tickvals": [0, 25, 50, 75, 100],
                    "ticktext": ["0%", "25%", "50%", "75%", "100%"],
                },
                "bar": {"color": color, "thickness": 0.30},
                "bgcolor": "#F1F5F9",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33], "color": "rgba(16, 185, 129, 0.12)"},
                    {"range": [33, 60], "color": "rgba(245, 158, 11, 0.12)"},
                    {"range": [60, 100], "color": "rgba(239, 68, 68, 0.12)"},
                ],
                "threshold": {
                    "line": {"color": "#2563EB", "width": 4},
                    "thickness": 0.85,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=25, r=25, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def sidebar_navigation():
    st.sidebar.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;padding:4px 6px 20px;">
          <div style="width:44px;height:44px;border-radius:14px;
                      background:linear-gradient(135deg,#2563EB,#8B5CF6);
                      display:flex;align-items:center;justify-content:center;
                      font-weight:900;color:#fff;font-size:20px;
                      box-shadow: 0 4px 14px rgba(37,99,235,.35);">👥</div>
          <div>
            <div style="font-size:18px;font-weight:900;color:#fff;letter-spacing:-.02em;">AttritionIQ</div>
            <div style="font-size:11.5px;color:#94B3D6;font-weight:600;">AI Turnover Intelligence</div>
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

    st.sidebar.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div style="font-size:11px;font-weight:800;letter-spacing:.08em;'
        'text-transform:uppercase;color:#8FA8C8;margin-bottom:8px;">Champion ML Engine</div>',
        unsafe_allow_html=True,
    )
    b = get_bundle()
    st.sidebar.markdown(
        f"""
        <div style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.14);
                    border-radius:16px;padding:16px 18px;backdrop-filter:blur(10px);">
          <div style="display:flex;align-items:center;justify-content:space-between;">
            <div style="font-size:13.5px;color:#fff;font-weight:800;">Support Vector Machine</div>
            <span style="background:#10B98130;color:#34D399;font-size:10px;font-weight:800;padding:2px 8px;border-radius:999px;">ACTIVE</span>
          </div>
          <div style="font-size:11.5px;color:#94B3D6;margin-top:2px;">RBF Kernel · Balanced Loss</div>
          <div style="display:flex;gap:14px;margin-top:14px;">
            <div>
              <div style="font-size:10px;color:#94A3B8;font-weight:700;">TEST F1</div>
              <div style="font-size:17px;font-weight:900;color:#fff;">{b['metrics']['f1']:.2f}</div>
            </div>
            <div>
              <div style="font-size:10px;color:#94A3B8;font-weight:700;">ROC-AUC</div>
              <div style="font-size:17px;font-weight:900;color:#fff;">{b['metrics']['roc_auc']:.2f}</div>
            </div>
            <div>
              <div style="font-size:10px;color:#94A3B8;font-weight:700;">THRESHOLD</div>
              <div style="font-size:17px;font-weight:900;color:#60A5FA;">{b['threshold']:.2f}</div>
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

    # Modern Enterprise Hero Banner
    st.markdown(
        """
        <div class="hero-card">
          <div class="hero-badge">
            <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#34D399;margin-right:2px;"></span>
            AttritionIQ Inference Engine v2.5
          </div>
          <div class="hero-title">Individual Employee Attrition Predictor</div>
          <div class="hero-sub">
            Real-time explainable AI decision support system. Select an employee from the directory or customize workplace parameters
            to quantify turnover probability, diagnose underlying SHAP drivers, and formulate prescriptive retention interventions.
          </div>
          <div class="hero-stat-row">
            <div class="hero-stat-pill"><span class="num">SVM (RBF)</span><span class="lbl">Champion Model</span></div>
            <div class="hero-stat-pill"><span class="num">0.817</span><span class="lbl">ROC-AUC Power</span></div>
            <div class="hero-stat-pill"><span class="num">33.0%</span><span class="lbl">Optimal Decision Cutoff</span></div>
            <div class="hero-stat-pill"><span class="num">Sub-Second</span><span class="lbl">SHAP Kernel Attribution</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Clean Directory Selector
    st.markdown('<div class="section-label">Step 1: Select Employee Profile From Directory</div>', unsafe_allow_html=True)
    selected_emp_id = st.selectbox(
        "Select Employee",
        options=list(EMPLOYEE_DIRECTORY.keys()),
        format_func=lambda k: EMPLOYEE_DIRECTORY[k]["label"],
        index=0,
        label_visibility="collapsed",
    )

    emp_meta = EMPLOYEE_DIRECTORY[selected_emp_id]
    emp_vals = emp_meta["data"]

    def val(k, default):
        if emp_vals and k in emp_vals:
            return emp_vals[k]
        return default

    # 2. Render Employee Profile Card
    if selected_emp_id != "CUSTOM":
        st.markdown(
            f"""
            <div class="emp-card">
              <div class="emp-avatar">{emp_meta['avatar']}</div>
              <div class="emp-info">
                <div class="emp-name">
                  {emp_meta['name']}
                  <span class="emp-id-badge">{emp_meta['id']}</span>
                </div>
                <div class="emp-role">{emp_meta['role']} · Department of {emp_meta['department']}</div>
                <div class="emp-summary">{emp_meta['summary']}</div>
                <div class="emp-chips">
                  <span class="emp-chip">👤 Age {val('Age', 35)}</span>
                  <span class="emp-chip">💵 ${val('MonthlyIncome', 5000):,}/month</span>
                  <span class="emp-chip">🏛️ {val('YearsAtCompany', 3)} Years Tenure</span>
                  <span class="emp-chip">⏰ Overtime: {val('OverTime', 'No')}</span>
                  <span class="emp-chip">🚗 Commute: {val('DistanceFromHome', 5)} Miles</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Clean Input Form Organized in Modern Tabs
    st.markdown('<div class="section-label">Step 2: Employee Attributes & Workplace Parameters</div>', unsafe_allow_html=True)

    with st.form("employee_assessment_form"):
        tab_demo, tab_role, tab_tenure = st.tabs([
            "📋 1. Personal & Commute",
            "💼 2. Role, Pay & Schedule",
            "⏱️ 3. Tenure & Satisfaction",
        ])

        with tab_demo:
            d_r1_1, d_r1_2, d_r1_3 = st.columns(3)
            with d_r1_1:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">👤 Employee Age <span style="font-size:11.5px;color:#64748B;font-weight:500;">(Years)</span></div>', unsafe_allow_html=True)
                age = st.number_input("Employee Age", 18, 65, int(val("Age", 35)), label_visibility="collapsed")
            with d_r1_2:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">⚧️ Gender</div>', unsafe_allow_html=True)
                gender_opts = ["Male", "Female"]
                gender = st.selectbox("Gender", gender_opts, index=gender_opts.index(val("Gender", "Male")), label_visibility="collapsed")
            with d_r1_3:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">💍 Marital Status</div>', unsafe_allow_html=True)
                mar_opts = ["Single", "Married", "Divorced"]
                marital = st.selectbox("Marital Status", mar_opts, index=mar_opts.index(val("MaritalStatus", "Single")), label_visibility="collapsed")

            d_r2_1, d_r2_2, d_r2_3 = st.columns(3)
            with d_r2_1:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">🚗 Commute Distance <span style="font-size:11.5px;color:#64748B;font-weight:500;">(Miles from home)</span></div>', unsafe_allow_html=True)
                distance = st.number_input("Commute Distance", 1, 35, int(val("DistanceFromHome", 8)), label_visibility="collapsed")
            with d_r2_2:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">🎓 Highest Education Level</div>', unsafe_allow_html=True)
                edu_opts = [
                    "1 - Below College / High School",
                    "2 - Associate Degree / College",
                    "3 - Bachelor Degree",
                    "4 - Master Degree",
                    "5 - Doctorate / PhD",
                ]
                edu_idx = min(max(int(val("Education", 3)) - 1, 0), 4)
                edu_sel = st.selectbox("Education Level", edu_opts, index=edu_idx, label_visibility="collapsed")
                education = int(edu_sel.split(" - ")[0])
            with d_r2_3:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">📚 Field of Education <span style="font-size:11.5px;color:#64748B;font-weight:500;">(Degree Major)</span></div>', unsafe_allow_html=True)
                edu_field_opts = ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]
                edu_field = st.selectbox("Field of Education", edu_field_opts, index=edu_field_opts.index(val("EducationField", "Life Sciences")), label_visibility="collapsed")

        with tab_role:
            r_r1_1, r_r1_2, r_r1_3 = st.columns(3)
            with r_r1_1:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">👔 Job Role / Title</div>', unsafe_allow_html=True)
                role_opts = [
                    "Sales Executive", "Research Scientist", "Laboratory Technician",
                    "Manufacturing Director", "Healthcare Representative", "Manager",
                    "Sales Representative", "Research Director", "Human Resources",
                ]
                role = st.selectbox("Job Role", role_opts, index=role_opts.index(val("JobRole", "Sales Executive")), label_visibility="collapsed")
            with r_r1_2:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">🏷️ Department</div>', unsafe_allow_html=True)
                dept_opts = ["Sales", "Research & Development", "Human Resources"]
                dept = st.selectbox("Department", dept_opts, index=dept_opts.index(val("Department", "Sales")), label_visibility="collapsed")
            with r_r1_3:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">⏰ Works Overtime? <span style="font-size:11.5px;color:#DC2626;font-weight:700;">(Major Churn Driver)</span></div>', unsafe_allow_html=True)
                ot_opts = ["Yes", "No"]
                overtime = st.selectbox("Works Overtime", ot_opts, index=ot_opts.index(val("OverTime", "Yes")), label_visibility="collapsed")

            r_r2_1, r_r2_2, r_r2_3 = st.columns(3)
            with r_r2_1:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">💵 Monthly Salary <span style="font-size:11.5px;color:#64748B;font-weight:500;">(USD $ / month)</span></div>', unsafe_allow_html=True)
                income = st.number_input("Monthly Salary", 1000, 25000, int(val("MonthlyIncome", 4800)), step=100, label_visibility="collapsed")
            with r_r2_2:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">📊 Annual Salary Hike <span style="font-size:11.5px;color:#64748B;font-weight:500;">(%)</span></div>', unsafe_allow_html=True)
                hike = st.number_input("Percent Salary Hike", 10, 30, int(val("PercentSalaryHike", 11)), label_visibility="collapsed")
            with r_r2_3:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">✈️ Business Travel Frequency</div>', unsafe_allow_html=True)
                travel_opts = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
                travel_display = {"Travel_Rarely": "Travels Rarely", "Travel_Frequently": "Travels Frequently", "Non-Travel": "No Travel"}
                travel = st.selectbox("Business Travel Frequency", travel_opts, index=travel_opts.index(val("BusinessTravel", "Travel_Rarely")), format_func=lambda x: travel_display[x], label_visibility="collapsed")

            r_r3_1, r_r3_2, r_r3_3 = st.columns(3)
            with r_r3_1:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">🎖️ Job Seniority Band <span style="font-size:11.5px;color:#64748B;font-weight:500;">(Level 1 to 5)</span></div>', unsafe_allow_html=True)
                job_level = st.slider("Job Seniority Band", 1, 5, int(val("JobLevel", 2)), label_visibility="collapsed")
            with r_r3_2:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">📈 Stock Option Equity Level <span style="font-size:11.5px;color:#64748B;font-weight:500;">(0=None to 3=Max)</span></div>', unsafe_allow_html=True)
                stock = st.slider("Stock Option Equity Level", 0, 3, int(val("StockOptionLevel", 0)), label_visibility="collapsed")
            with r_r3_3:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">⭐ Performance Review Rating</div>', unsafe_allow_html=True)
                perf_opts = ["3 - Excellent Performance", "4 - Outstanding Performance"]
                perf_idx = 0 if int(val("PerformanceRating", 3)) == 3 else 1
                perf_sel = st.selectbox("Performance Review Rating", perf_opts, index=perf_idx, label_visibility="collapsed")
                perf = int(perf_sel.split(" - ")[0])

        with tab_tenure:
            t_r1_1, t_r1_2, t_r1_3 = st.columns(3)
            with t_r1_1:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">🏛️ Tenure at Company <span style="font-size:11.5px;color:#64748B;font-weight:500;">(Years)</span></div>', unsafe_allow_html=True)
                years_co = st.number_input("Tenure at Company", 0, 40, int(val("YearsAtCompany", 4)), label_visibility="collapsed")
            with t_r1_2:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">🎯 Years in Current Role</div>', unsafe_allow_html=True)
                years_role = st.number_input("Years in Current Role", 0, 20, int(val("YearsInCurrentRole", 2)), label_visibility="collapsed")
            with t_r1_3:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">💼 Total Career Experience <span style="font-size:11.5px;color:#64748B;font-weight:500;">(Years)</span></div>', unsafe_allow_html=True)
                total_work = st.number_input("Total Career Experience", 0, 45, int(val("TotalWorkingYears", 8)), label_visibility="collapsed")

            t_r2_1, t_r2_2, t_r2_3 = st.columns(3)
            with t_r2_1:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">🚀 Years Since Last Promotion</div>', unsafe_allow_html=True)
                last_promo = st.number_input("Years Since Last Promotion", 0, 20, int(val("YearsSinceLastPromotion", 2)), label_visibility="collapsed")
            with t_r2_2:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">🤝 Years With Current Manager</div>', unsafe_allow_html=True)
                years_mgr = st.number_input("Years With Current Manager", 0, 20, int(val("YearsWithCurrManager", 2)), label_visibility="collapsed")
            with t_r2_3:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">🏢 Prior Companies Worked</div>', unsafe_allow_html=True)
                num_cos = st.number_input("Previous Companies Worked", 0, 10, int(val("NumCompaniesWorked", 8)), label_visibility="collapsed")

            t_r3_1, t_r3_2, t_r3_3 = st.columns(3)
            with t_r3_1:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">📖 Training Sessions Last Year</div>', unsafe_allow_html=True)
                training = st.number_input("Training Sessions Last Year", 0, 8, int(val("TrainingTimesLastYear", 1)), label_visibility="collapsed")
            with t_r3_2:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">😊 Job Satisfaction Rating</div>', unsafe_allow_html=True)
                sat_opts = ["1 - Very Dissatisfied", "2 - Dissatisfied", "3 - Satisfied", "4 - Very Satisfied"]
                sat_idx = min(max(int(val("JobSatisfaction", 2)) - 1, 0), 3)
                sat_sel = st.selectbox("Job Satisfaction Rating", sat_opts, index=sat_idx, label_visibility="collapsed")
                sat = int(sat_sel.split(" - ")[0])
            with t_r3_3:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">🏢 Workplace Environment Satisfaction</div>', unsafe_allow_html=True)
                env_opts = ["1 - Low Satisfaction", "2 - Medium", "3 - High", "4 - Very High"]
                env_idx = min(max(int(val("EnvironmentSatisfaction", 2)) - 1, 0), 3)
                env_sel = st.selectbox("Work Environment Satisfaction", env_opts, index=env_idx, label_visibility="collapsed")
                env_sat = int(env_sel.split(" - ")[0])

            t_r4_1, t_r4_2, t_r4_3 = st.columns(3)
            with t_r4_1:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">⚖️ Work-Life Balance Rating</div>', unsafe_allow_html=True)
                wlb_opts = ["1 - Poor (Burnout Risk)", "2 - Below Average", "3 - Good / Balanced", "4 - Excellent"]
                wlb_idx = min(max(int(val("WorkLifeBalance", 1)) - 1, 0), 3)
                wlb_sel = st.selectbox("Work-Life Balance Rating", wlb_opts, index=wlb_idx, label_visibility="collapsed")
                wlb = int(wlb_sel.split(" - ")[0])
            with t_r4_2:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">🔥 Job Involvement Rating</div>', unsafe_allow_html=True)
                involve_opts = ["1 - Low", "2 - Medium", "3 - High", "4 - Very High"]
                involve_idx = min(max(int(val("JobInvolvement", 3)) - 1, 0), 3)
                involve_sel = st.selectbox("Job Involvement Rating", involve_opts, index=involve_idx, label_visibility="collapsed")
                involvement = int(involve_sel.split(" - ")[0])
            with t_r4_3:
                st.markdown('<div style="font-size:13.5px;font-weight:800;color:#0F172A;margin-bottom:4px;">🤝 Team Relationship Satisfaction</div>', unsafe_allow_html=True)
                rel_opts = ["1 - Low", "2 - Medium", "3 - High", "4 - Very High"]
                rel_idx = min(max(int(val("RelationshipSatisfaction", 1)) - 1, 0), 3)
                rel_sel = st.selectbox("Team Relationship Satisfaction", rel_opts, index=rel_idx, label_visibility="collapsed")
                rel_sat = int(rel_sel.split(" - ")[0])

        daily_rate = int(val("DailyRate", max(100, int(income / 22 * 3.5))))
        hourly_rate = int(val("HourlyRate", max(15, int(income / 160))))
        monthly_rate = int(val("MonthlyRate", max(2000, int(income * 2.8))))

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡ Compute Attrition Probability & AI Diagnosis", use_container_width=True)

    current_sample = {
        "Age": age, "BusinessTravel": travel, "DailyRate": daily_rate, "Department": dept,
        "DistanceFromHome": distance, "Education": education, "EducationField": edu_field,
        "EnvironmentSatisfaction": env_sat, "Gender": gender, "HourlyRate": hourly_rate,
        "JobInvolvement": involvement, "JobLevel": job_level, "JobRole": role,
        "JobSatisfaction": sat, "MaritalStatus": marital, "MonthlyIncome": income,
        "MonthlyRate": monthly_rate, "NumCompaniesWorked": num_cos, "OverTime": overtime,
        "PercentSalaryHike": hike, "PerformanceRating": perf,
        "RelationshipSatisfaction": rel_sat, "StockOptionLevel": stock,
        "TotalWorkingYears": total_work, "TrainingTimesLastYear": training,
        "WorkLifeBalance": wlb, "YearsAtCompany": years_co,
        "YearsInCurrentRole": years_role, "YearsSinceLastPromotion": last_promo,
        "YearsWithCurrManager": years_mgr,
    }

    # 4. Execute Prediction
    with st.spinner("Calculating calibrated SVM attrition prediction..."):
        prob = predict_probability(b, current_sample)
        st.session_state["last_sample"] = current_sample
        st.session_state["last_prob"] = prob

    is_leaver = prob >= threshold
    tier = risk_level(prob)
    c_color = RISK[tier]
    person_name = emp_meta["name"] if selected_emp_id != "CUSTOM" else "Assessed Employee"

    st.markdown('<div class="section-label">Step 3: Prediction Results & AI Diagnosis</div>', unsafe_allow_html=True)

    # Clean Verdict Banner
    if is_leaver:
        st.markdown(
            f"""
            <div class="verdict-box high">
              <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px;">
                <div>
                  <div style="font-size:22px;font-weight:900;color:#991B1B;display:flex;align-items:center;gap:10px;">
                    🚨 PREDICTION VERDICT: HIGH ATTRITION RISK (LIKELY TO LEAVE)
                  </div>
                  <div style="font-size:14px;color:#7F1D1D;margin-top:6px;line-height:1.6;">
                    <b>{person_name}</b> has a predicted turnover probability of <b style="font-size:16px;">{prob:.1%}</b>, which exceeds the calibrated decision threshold (<b>{threshold:.1%}</b>).
                    Active retention measures are strongly recommended to mitigate employee turnover.
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
              <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px;">
                <div>
                  <div style="font-size:22px;font-weight:900;color:#166534;display:flex;align-items:center;gap:10px;">
                    ✅ PREDICTION VERDICT: LOW ATTRITION RISK (LIKELY TO STAY)
                  </div>
                  <div style="font-size:14px;color:#14532D;margin-top:6px;line-height:1.6;">
                    <b>{person_name}</b> has a predicted turnover probability of <b style="font-size:16px;">{prob:.1%}</b>, safely below the decision threshold (<b>{threshold:.1%}</b>).
                    Employee demonstrates healthy retention stability indicators. Standard engagement practices are sufficient.
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
    rec_text = "Intervention Urgent" if is_leaver else "Standard Monitoring"
    k_cols[3].markdown(kpi_card("HR Action Protocol", rec_text, "Action Required", "🛡️", VIOLET, "#F5F3FF"), unsafe_allow_html=True)

    # Modern Gauge Meter Panel
    st.markdown('<div class="panel" style="margin-top:18px;">', unsafe_allow_html=True)
    g_col1, g_col2 = st.columns([1.4, 1])
    with g_col1:
        st.markdown('<div class="panel-title">🎯 Calibrated Probability Gauge</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Optimal decision boundary calibrated at 33.0% (blue threshold needle).</div>', unsafe_allow_html=True)
        st.plotly_chart(render_gauge_chart(prob, threshold, c_color), use_container_width=True, config={"displayModeBar": False})
    with g_col2:
        st.markdown('<div class="panel-title">📊 Risk Tier Stratification</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="margin-top:14px;display:flex;flex-direction:column;gap:10px;">
              <div style="display:flex;justify-content:space-between;padding:8px 12px;background:{'#FEE2E2' if tier == 'Critical' else '#F8FAFC'};border-radius:10px;border-left:4px solid #991B1B;">
                <span style="font-weight:700;font-size:13px;">Critical Risk Tier</span>
                <span style="font-weight:800;font-size:13px;color:#991B1B;">80% – 100%</span>
              </div>
              <div style="display:flex;justify-content:space-between;padding:8px 12px;background:{'#FEE2E2' if tier == 'High' else '#F8FAFC'};border-radius:10px;border-left:4px solid #EF4444;">
                <span style="font-weight:700;font-size:13px;">High Risk Tier</span>
                <span style="font-weight:800;font-size:13px;color:#EF4444;">55% – 80%</span>
              </div>
              <div style="display:flex;justify-content:space-between;padding:8px 12px;background:{'#FEF3C7' if tier == 'Medium' else '#F8FAFC'};border-radius:10px;border-left:4px solid #F59E0B;">
                <span style="font-weight:700;font-size:13px;">Moderate Risk Tier</span>
                <span style="font-weight:800;font-size:13px;color:#F59E0B;">33% – 55%</span>
              </div>
              <div style="display:flex;justify-content:space-between;padding:8px 12px;background:{'#DCFCE7' if tier == 'Low' else '#F8FAFC'};border-radius:10px;border-left:4px solid #10B981;">
                <span style="font-weight:700;font-size:13px;">Safe Retention Tier</span>
                <span style="font-weight:800;font-size:13px;color:#10B981;">0% – 33%</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # 5. SHAP Breakdown with Explicit Human-Friendly Names & Current Values
    exp_col, rec_col = st.columns([1.3, 1])

    with exp_col:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">🧠 SHAP Explainability: Top Prediction Drivers</div>
              <div class="panel-sub">Local SHAP kernel attribution showing exactly which factors push toward departure or retention:</div>
            """,
            unsafe_allow_html=True,
        )
        try:
            with st.spinner("Decomposing SHAP contributions..."):
                factors, _ = explain_prediction(b, current_sample, top_n=6, n_samples=100)

            clean_labels = []
            vals = []
            table_rows = []

            for raw_feature, shap_val in factors:
                meta = FEATURE_METADATA.get(raw_feature, {})
                display_name = meta.get("name", raw_feature)
                icon = meta.get("icon", "🔹")
                raw_val = current_sample.get(raw_feature, "")
                formatted_val = meta.get("format", lambda x: str(x))(raw_val)

                clean_label = f"{display_name} ({formatted_val})"
                clean_labels.append(clean_label)
                vals.append(shap_val)

                impact_text = f"🔴 Increases Attrition (+{abs(shap_val):.1%})" if shap_val > 0 else f"🟢 Promotes Retention (-{abs(shap_val):.1%})"
                table_rows.append({
                    "Attribute / Factor": f"{icon} {display_name}",
                    "Employee's Current Value": formatted_val,
                    "Impact on Turnover": impact_text,
                })

            mx = max(abs(v) for v in vals) if vals else 1.0

            fig_shap = go.Figure(
                go.Bar(
                    y=clean_labels,
                    x=vals,
                    orientation="h",
                    marker=dict(
                        color=[RED if v >= 0 else EMERALD for v in vals],
                        cornerradius=6,
                    ),
                    text=[f"{v:+.3f}" for v in vals],
                    textposition="outside",
                    textfont=dict(size=11, color=INK, weight=700),
                    hovertemplate="%{y}: SHAP %{x:+.3f}<extra></extra>",
                )
            )
            fig_shap.update_xaxes(range=[-mx * 1.35, mx * 1.35], title="SHAP Attribution (Red = Increases Churn, Green = Increases Retention)")
            fig_shap.update_layout(height=320)
            st.plotly_chart(style_fig(fig_shap), use_container_width=True, config={"displayModeBar": False})

            st.markdown('<div style="font-size:13px;font-weight:800;color:#334155;margin:12px 0 6px;">Detailed Factor Breakdown:</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

        except Exception as e:
            st.warning(f"SHAP explanation note: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with rec_col:
        st.markdown(
            f"""
            <div class="panel">
              <div class="panel-title">🛡️ Prescriptive Retention Action Plan</div>
              <div class="panel-sub">Automated strategic HR interventions tailored for <b>{person_name}</b>:</div>
              <div style="margin-top:16px;">
            """,
            unsafe_allow_html=True,
        )

        recs = []
        if overtime == "Yes":
            recs.append(("p1", "Priority 1 · Urgent", "Overtime Workload Ceiling", "Employee is logging regular overtime (+2.3x churn multiplier). Implement an overtime cap and redistribute project milestones."))
        if sat <= 2:
            recs.append(("p1", "Priority 1 · Urgent", "Confidential Stay Interview", "Job satisfaction is critically low. Schedule a confidential 1-on-1 retention interview within 7 business days to resolve friction."))
        if income < 4500:
            recs.append(("p2", "Priority 2 · High", "Compensation Equity Review", "Monthly salary is below the peer role median. Review market benchmarks and consider an off-cycle salary adjustment or spot bonus."))
        if wlb <= 2:
            recs.append(("p2", "Priority 2 · High", "Work-Life Balance Initiative", "Employee reports acute burnout risk. Introduce flexible working hours or partial hybrid remote scheduling."))
        if distance > 12:
            recs.append(("p3", "Priority 3 · Medium", "Commute & Remote Flexibility", "Commute distance is >12 miles. Provide transit subsidy or grant 2–3 remote work days per week."))
        if last_promo >= 3:
            recs.append(("p3", "Priority 3 · Medium", "Career Progression Review", "Over 3 years without promotion. Formulate a 6-month promotion readiness roadmap with clear deliverables."))
        if not recs:
            recs.append(("p3", "Routine Check-In", "Retention Maintenance", "Employee exhibits strong engagement and compensation alignment. Maintain regular quarterly milestone check-ins."))

        for p_class, p_lbl, title, desc in recs:
            st.markdown(
                f"""
                <div class="rec-card">
                  <span class="rec-badge {p_class}">{p_lbl}</span>
                  <div class="rec-text">
                    <b>{title}:</b> {desc}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
        <div class="hero-card">
          <div class="hero-badge">📁 Batch Workforce Analytics</div>
          <div class="hero-title">Workforce Batch Attrition Scoring</div>
          <div class="hero-sub">
            Upload an entire company or department CSV roster to predict attrition risk across all employees simultaneously.
            Quantify financial turnover exposure, stratify risk tiers, and export scored rosters.
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
        <div class="hero-card">
          <div class="hero-badge">🔄 Policy Simulation Sandbox</div>
          <div class="hero-title">Interactive Retention Intervention Simulator</div>
          <div class="hero-sub">
            Select any employee from the directory and simulate targeted HR retention policies.
            Quantify the turnover probability reduction, risk tier migration, and return on investment (ROI) before executing organizational changes.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Base Employee Selection
    sim_emp_id = st.selectbox(
        "Select Employee Profile to Simulate Interventions:",
        options=[k for k in EMPLOYEE_DIRECTORY.keys() if k != "CUSTOM"],
        format_func=lambda k: EMPLOYEE_DIRECTORY[k]["label"],
        index=0,
    )
    sim_meta = EMPLOYEE_DIRECTORY[sim_emp_id]
    base_sample = dict(sim_meta["data"])

    base_prob = predict_probability(b, base_sample)
    base_tier = risk_level(base_prob)

    # Render summary card
    st.markdown(
        f"""
        <div class="emp-card">
          <div class="emp-avatar">{sim_meta['avatar']}</div>
          <div class="emp-info">
            <div class="emp-name">{sim_meta['name']} <span class="emp-id-badge">{sim_meta['id']}</span></div>
            <div class="emp-role">{sim_meta['role']} · {sim_meta['department']}</div>
            <div class="emp-summary">{sim_meta['summary']}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Simulate Proactive HR Policies</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        st.markdown("<b>Workload & Flexibility Interventions:</b>", unsafe_allow_html=True)
        eliminate_ot = st.checkbox(
            "Eliminate Mandatory Overtime (OverTime: Yes ➔ No)",
            value=(base_sample.get("OverTime") == "Yes"),
        )
        improve_wlb = st.slider(
            "Improve Work-Life Balance Rating (1=Poor to 4=Excellent)",
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
            "Grant Retention Stock Option Equity (Stock Option Level ➔ 2)",
            value=False,
        )

    # Apply simulated interventions
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
    st.markdown('<div class="section-label">Simulated Impact Analysis</div>', unsafe_allow_html=True)
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
        annotation_text=f"Decision Boundary ({threshold:.1%})",
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
            By reducing {sim_meta['name']}'s turnover probability from <b>{base_prob:.1%}</b> to <b>{sim_prob:.1%}</b>, the organization avoids an estimated
            <b>${est_turnover_cost:,.0f}</b> replacement expense for an intervention investment of approximately <b>${est_intervention_cost:,.0f}</b>.
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
    rate = df["Attrition"].eq("Yes").mean()

    st.markdown(
        f"""
        <div class="hero-card">
          <div class="hero-badge">📊 Historical Insights</div>
          <div class="hero-title">Workforce Analytics & Historical Exploration</div>
          <div class="hero-sub">
            Empirical baseline analysis on 1,470 employee records from the IBM HR Analytics benchmark dataset.
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
        <div class="hero-card">
          <div class="hero-badge">🧠 Model Validation Suite</div>
          <div class="hero-title">Machine Learning Model Evaluation & Benchmark</div>
          <div class="hero-sub">
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