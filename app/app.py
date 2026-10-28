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
BG = "#F1F5F9"
PRIMARY = "#2563EB"
PRIMARY_DK = "#1E40AF"
EMERALD = "#10B981"
RED = "#EF4444"
AMBER = "#F59E0B"
VIOLET = "#8B5CF6"

RISK = {"Low": EMERALD, "Medium": AMBER, "High": RED, "Critical": "#7F1D1D"}

PRESETS = {
    "Custom (Manual Input)": None,
    "High Risk (Sales Rep, Frequent Travel, Overtime)": {
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
    "Low Risk (Senior Tech Director, High Satisfaction)": {
        "Age": 45, "BusinessTravel": "Travel_Rarely", "DailyRate": 900, "Department": "Research & Development",
        "DistanceFromHome": 3, "Education": 4, "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 4, "Gender": "Female", "HourlyRate": 75,
        "JobInvolvement": 4, "JobLevel": 3, "JobRole": "Manufacturing Director",
        "JobSatisfaction": 4, "MaritalStatus": "Married", "MonthlyIncome": 12500,
        "MonthlyRate": 16000, "NumCompaniesWorked": 1, "OverTime": "No",
        "PercentSalaryHike": 18, "PerformanceRating": 3,
        "RelationshipSatisfaction": 4, "StockOptionLevel": 2,
        "TotalWorkingYears": 18, "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 3, "YearsAtCompany": 12,
        "YearsInCurrentRole": 8, "YearsSinceLastPromotion": 2,
        "YearsWithCurrManager": 7,
    },
    "Average Employee Profile (R&D Scientist)": {
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
}

st.set_page_config(
    page_title="AttritionIQ | Employee Attrition Analytics",
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
.stApp { background: #F1F5F9; }
.block-container { padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1240px; }

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0B1B33 0%, #12294C 55%, #17335F 100%);
  border-right: 1px solid rgba(255,255,255,.06);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
div[role="radiogroup"] { gap: 4px; }
div[role="radiogroup"] label {
  padding: 0.5rem 0.75rem !important; border-radius: 10px; cursor: pointer;
  transition: background .15s ease;
}
div[role="radiogroup"] label:hover { background: rgba(255,255,255,.07); }
div[role="radiogroup"] label:has(input:checked) {
  background: rgba(37,99,235,.28);
  box-shadow: inset 0 0 0 1px rgba(96,165,250,.45);
}
div[role="radiogroup"] p, div[role="radiogroup"] [data-testid="stMarkdownContainer"] p { font-weight: 500 !important; }

/* ---------- Buttons ---------- */
.stButton > button, [data-testid="stFormSubmitButton"] > button, [data-testid="stWidgetLabel"] {
  border-radius: 10px; font-weight: 600;
}
.stButton > button, [data-testid="stFormSubmitButton"] > button {
  border: none; background: linear-gradient(135deg,#2563EB,#1E40AF); color: #fff;
  padding: 0.62em 1.4em; transition: all .18s; box-shadow: 0 6px 16px rgba(37,99,235,.30);
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
  transform: translateY(-1px); box-shadow: 0 10px 24px rgba(37,99,235,.38); color: #fff;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #E2E8F0; }
.stTabs [data-baseweb="tab"] {
  border-radius: 12px 12px 0 0; padding: 9px 20px; font-weight: 600;
  background: #E8EFFA; color: #475569;
}
.stTabs [aria-selected="true"] { background: #2563EB !important; color: #fff !important; }

/* ---------- Inputs ---------- */
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input, textarea {
  border-radius: 10px !important; border: 1px solid #E2E8F0;
}
[data-baseweb="select"] > div { border-radius: 10px !important; border: 1px solid #E2E8F0; }
[data-testid="stSlider"] [role="slider"] { background: #2563EB !important; }

/* ---------- Results ---------- */
.hero {
  background: linear-gradient(135deg,#0B1B33 0%, #14355C 55%, #1E4E88 100%);
  border-radius: 20px; padding: 30px 34px; margin-bottom: 20px; position: relative; overflow: hidden;
}
.hero::after {
  content: ""; position: absolute; right: -80px; top: -80px; width: 320px; height: 320px;
  border-radius: 50%; background: radial-gradient(circle, rgba(37,99,235,.35), transparent 70%);
}
.hero-title { font-size: 34px; font-weight: 900; color: #fff; letter-spacing: -.02em; }
.hero-sub { font-size: 14.5px; color: #B6CBE4; margin-top: 6px; max-width: 640px; }
.hero-chip {
  display: inline-flex; align-items: baseline; gap: 8px; background: rgba(255,255,255,.10);
  border: 1px solid rgba(255,255,255,.16); border-radius: 12px; padding: 10px 16px; backdrop-filter: blur(4px);
}
.hero-chip .n { font-size: 22px; font-weight: 800; color: #fff; }
.hero-chip .l { font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: #94B3D6; }

.panel {
  background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 18px;
  padding: 20px 22px; box-shadow: 0 4px 20px rgba(15,23,42,.06);
  margin-bottom: 18px;
}
.panel-title { font-size: 15px; font-weight: 800; color: #0F172A; }
.panel-sub { font-size: 12.5px; color: #64748B; margin-top: 2px; }

.kpi {
  background:#fff; border:1px solid #E2E8F0; border-radius:18px; padding:18px 20px;
  box-shadow:0 4px 20px rgba(15,23,42,.05); position:relative; overflow:hidden;
}
.kpi .accent { position:absolute; left:0; top:0; bottom:0; width:5px; }
.kpi .icon {
  width:38px; height:38px; border-radius:11px; display:flex; align-items:center; justify-content:center;
  font-size:18px; margin-bottom:10px;
}
.kpi .lbl { font-size:11px; font-weight:700; letter-spacing:.07em; text-transform:uppercase; color:#64748B; }
.kpi .val { font-size:26px; font-weight:800; color:#0F172A; margin-top:2px; letter-spacing:-.01em; }
.kpi .sub { font-size:12px; color:#94A3B8; margin-top:2px; }

.pill {
  display:inline-flex; align-items:center; gap:6px; padding:5px 14px; border-radius:999px;
  font-size:13px; font-weight:700;
}
.tag {
  display:inline-block; padding:3px 10px; border-radius:8px; font-size:11px; font-weight:700;
  letter-spacing:.05em; text-transform:uppercase; background:#E8EFFA; color:#1E40AF;
}
.section-label { font-size:12px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:#94A3B8; margin:22px 0 12px; }

.gauge-track { display:flex; gap:5px; margin-top:10px; }
.gauge-seg { flex:1; height:13px; border-radius:4px; background:#E6EDF5; transition: background .4s; }

.mono { font-family:'JetBrains Mono', monospace; }
.formula {
  font-family:'JetBrains Mono', monospace; font-size:13.5px; background:#F0F6FF;
  border:1px solid #DBEAFE; border-radius:12px; padding:12px 16px; display:inline-block; color:#1E40AF;
}
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
    return {"Support Vector Machine": "SVM", "Logistic Regression": "LR",
            "Random Forest": "RF", "Decision Tree": "DT",
            "K-Nearest Neighbors": "KNN"}.get(name, name)


def style_fig(fig, height=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=INK, size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11)),
        hoverlabel=dict(font=dict(family="Inter", size=12)),
    )
    fig.update_xaxes(gridcolor="#E8EEF5", zeroline=False, showline=False, tickfont=dict(color=MUT))
    fig.update_yaxes(gridcolor="#E8EEF5", zeroline=False, tickfont=dict(color=MUT))
    if height:
        fig.update_layout(height=height)
    return fig


def kpi(name, val, sub, icon, accent, icon_bg):
    return f"""
    <div class="kpi">
      <div class="accent" style="background:{accent};"></div>
      <div class="icon" style="background:{icon_bg};">{icon}</div>
      <div class="lbl">{name}</div>
      <div class="val">{val}</div>
      <div class="sub">{sub}</div>
    </div>"""


def risk_pill(level):
    c = RISK[level]
    return f'<span class="pill" style="background:{c}15;color:{c};border:1px solid {c}40;">{level} risk</span>'


def panel_open(title, sub=None):
    s = f'<div class="panel"><div class="panel-title">{title}</div>'
    if sub:
        s += f'<div class="panel-sub">{sub}</div>'
    st.markdown(s, unsafe_allow_html=True)


def panel_close():
    st.markdown("</div>", unsafe_allow_html=True)


def seg_gauge(fraction, color):
    segs = 12
    filled = int(round(max(0.0, min(fraction, 1.0)) * segs))
    cells = "".join(
        f'<div class="gauge-seg" style="background:{color if i < filled else "#E6EDF5"};"></div>'
        for i in range(segs)
    )
    return f'<div class="gauge-track">{cells}</div>'


# ---------------------------------------------------------------------------
def sidebar():
    st.sidebar.markdown(
        '<div style="display:flex;align-items:center;gap:12px;padding:4px 6px 20px;">'
        '<div style="width:42px;height:42px;border-radius:12px;'
        'background:linear-gradient(135deg,#2563EB,#8B5CF6);display:flex;align-items:center;'
        'justify-content:center;font-weight:900;color:#fff;font-size:16px;">A</div>'
        '<div><div style="font-size:17px;font-weight:800;color:#fff;">AttritionIQ</div>'
        '<div style="font-size:11.5px;color:#8FA8C8;">HR Risk Analytics</div></div></div>',
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "nav",
        ["Dashboard", "Predict", "Models", "Explainability"],
        label_visibility="collapsed",
        format_func=lambda x: {
            "Dashboard": "  Overview",
            "Predict": "  Risk Assessment",
            "Models": "  Model Performance",
            "Explainability": "  Explainability",
        }[x],
    )
    st.sidebar.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div style="font-size:11px;font-weight:700;letter-spacing:.08em;'
        'text-transform:uppercase;color:#8FA8C8;margin-bottom:8px;">Active model</div>',
        unsafe_allow_html=True,
    )
    b = get_bundle()
    st.sidebar.markdown(
        f'<div style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);'
        f'border-radius:14px;padding:14px 16px;">'
        f'<div style="font-size:13px;color:#fff;font-weight:700;">{short_model(b["model_name"])} '
        f'<span style="color:#8FA8C8;font-weight:500;">· tuned RBF</span></div>'
        f'<div style="display:flex;gap:18px;margin-top:10px;">'
        f'<div><div style="font-size:10px;color:#8FA8C8;">F1</div>'
        f'<div style="font-size:16px;font-weight:800;color:#fff;">{b["metrics"]["f1"]:.2f}</div></div>'
        f'<div><div style="font-size:10px;color:#8FA8C8;">AUC</div>'
        f'<div style="font-size:16px;font-weight:800;color:#fff;">{b["metrics"]["roc_auc"]:.2f}</div></div>'
        f'<div><div style="font-size:10px;color:#8FA8C8;">THRESH</div>'
        f'<div style="font-size:16px;font-weight:800;color:#fff;">{b["threshold"]:.2f}</div></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f'<div style="position:fixed;bottom:18px;left:22px;right:20px;font-size:11px;color:#5F7CA3;'
        f'line-height:1.6;">IBM HR Analytics dataset · 1,470 employees<br>scikit-learn + SHAP · Pipeline {sys.version.split()[0]}</div>',
        unsafe_allow_html=True,
    )
    return page


def hero():
    df = get_data()
    rate = df["Attrition"].eq("Yes").mean()
    guilty = df["OverTime"].eq("Yes").mean()
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-title">Employee Attrition Analytics</div>
          <div class="hero-sub">Explainable binary-classification risk system. Five models benchmarked on
          the IBM HR dataset, best model selected by F1, every prediction explained with SHAP.</div>
          <div style="display:flex;gap:12px;margin-top:22px;flex-wrap:wrap;">
            <div class="hero-chip"><span class="n" style="color:#60A5FA;">{len(df):,}</span><span class="l">Employees</span></div>
            <div class="hero-chip"><span class="n" style="color:#F87171;">{rate:.1%}</span><span class="l">Attrition</span></div>
            <div class="hero-chip"><span class="n" style="color:#34D399;">{1-rate:.1%}</span><span class="l">Retention</span></div>
            <div class="hero-chip"><span class="n" style="color:#FBBF24;">{guilty:.0%}</span><span class="l">Work overtime</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
def page_dashboard():
    df = get_data()
    b = get_bundle()

    hero()

    c = st.columns(4)
    c[0].markdown(kpi("Workforce", f"{len(df):,}", "IBM HR records", "👥", PRIMARY, "#E8F0FE"), unsafe_allow_html=True)
    c[1].markdown(kpi("Attrition", f'{df["Attrition"].eq("Yes").mean():.1%}', "237 leavers of 1470", "📉", RED, "#FEE2E2"), unsafe_allow_html=True)
    c[2].markdown(kpi("Model", short_model(b["model_name"]), f'test F1 {b["metrics"]["f1"]:.2f}', "🧠", VIOLET, "#F3E8FF"), unsafe_allow_html=True)
    c[3].markdown(kpi("ROC-AUC", f"{b['metrics']['roc_auc']:.3f}", "ranking quality", "🎯", EMERALD, "#D1FAE5"), unsafe_allow_html=True)

    st.markdown('<div class="section-label">Attrition Drivers</div>', unsafe_allow_html=True)

    r1 = st.columns(2)

    with r1[0]:
        panel_open("Attrition distribution", "84:16 split illustrates why accuracy alone is misleading")
        counts = df["Attrition"].value_counts()
        fig = go.Figure(
            go.Pie(
                labels=["Stayed", "Left"],
                values=counts.values,
                hole=0.66,
                marker=dict(colors=[PRIMARY, "#F97316"]),
                textinfo="label+percent",
                textfont=dict(size=13, color="#fff"),
                hovertemplate="%{label}: %{value} employees (%{percent})<extra></extra>",
            )
        )
        fig.update_layout(
            annotations=[dict(text=f'<b>{counts["No"]:,}</b><br><span style="font-size:11px;color:#64748B">retained</span>', x=0.5, y=0.5, showarrow=False)]
        )
        st.plotly_chart(style_fig(fig, 300), use_container_width=True, config={"displayModeBar": False})
        panel_close()

    with r1[1]:
        panel_open("Attrition rate by Overtime", "overtime roughly doubles attrition risk")
        ot = df.groupby("OverTime")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)
        fig = go.Figure(
            go.Bar(
                x=ot.values,
                y=["No overtime", "Overtime"],
                orientation="h",
                marker=dict(color=[PRIMARY, RED], cornerradius=6),
                text=[f"{v:.1f}%" for v in ot.values],
                textposition="outside",
                textfont=dict(size=13, color=INK),
                hovertemplate="%{y}: %{x:.1f}% leave<extra></extra>",
            )
        )
        fig.update_xaxes(range=[0, max(ot.values) * 1.3])
        st.plotly_chart(style_fig(fig, 300), use_container_width=True, config={"displayModeBar": False})
        panel_close()

    r2 = st.columns(2)

    with r2[0]:
        panel_open("Attrition by Department", "sales carries the heaviest churn burden")
        ct = pd.crosstab(df["Department"], df["Attrition"])[["No", "Yes"]]
        ct = ct.loc[["Human Resources", "Research & Development", "Sales"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=ct.index, x=ct["No"] / ct.sum(axis=1) * 100, orientation="h",
                             name="Stayed", marker_color=PRIMARY, hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig.add_trace(go.Bar(y=ct.index, x=ct["Yes"] / ct.sum(axis=1) * 100, orientation="h",
                             name="Left", marker_color="#F97316", hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig.update_layout(barmode="stack", xaxis_title="share of department (%)")
        st.plotly_chart(style_fig(fig, 300), use_container_width=True, config={"displayModeBar": False})
        panel_close()

    with r2[1]:
        panel_open("Attrition by Work-Life Balance", "declining balance, climbing risk")
        wlb = df.groupby("WorkLifeBalance")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)
        fig = go.Figure(
            go.Scatter(x=wlb.index, y=wlb.values, mode="lines+markers+text",
                       line=dict(color=AMBER, width=3), marker=dict(size=9, color=AMBER),
                       text=[f"{v:.1f}%" for v in wlb.values], textposition="top center",
                       textfont=dict(size=11, color=INK),
                       hovertemplate="Balance %{x}: %{y:.1f}% leave<extra></extra>")
        )
        fig.update_layout(xaxis=dict(tickmode="array", tickvals=[1, 2, 3, 4]))
        st.plotly_chart(style_fig(fig, 300), use_container_width=True, config={"displayModeBar": False})
        panel_close()

    st.markdown('<div class="section-label">What Drives Attrition — Model View</div>', unsafe_allow_html=True)
    panel_open("Top drivers of attrition (permutation importance)", "shuffling these features degrades predictive F1 the most")
    top = pd.DataFrame(b["top_importance"])[::-1]
    fig = go.Figure(
        go.Bar(
            x=top["importance"], y=top["feature"], orientation="h",
            marker=dict(color=[VIOLET] * len(top), cornerradius=5),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="permutation importance")
    st.plotly_chart(style_fig(fig, 520), use_container_width=True, config={"displayModeBar": False})
    panel_close()


# ---------------------------------------------------------------------------
def prediction_form(preset_vals=None):
    vals = preset_vals or {}

    def get_val(col, default):
        return vals.get(col, default)

    with st.form("predict_form"):
        st.markdown(
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">'
            '<span class="tag">Lifestyle</span><span style="font-weight:800;font-size:15px;">Demographics & Satisfaction</span></div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 60, int(get_val("Age", 34)))
        distance = c1.number_input("Distance from home (mi)", 1, 29, int(get_val("DistanceFromHome", 7)))
        satisfaction = c1.slider("Job satisfaction", 1, 4, int(get_val("JobSatisfaction", 3)))
        income = c2.number_input("Monthly income (USD)", 1000, 20000, int(get_val("MonthlyIncome", 5600)), step=100)
        
        ot_options = ["No", "Yes"]
        ot_val = get_val("OverTime", "No")
        overtime = c2.selectbox("Overtime", ot_options, index=ot_options.index(ot_val) if ot_val in ot_options else 0)
        
        env_sat = c2.slider("Environment satisfaction", 1, 4, int(get_val("EnvironmentSatisfaction", 3)))
        
        mar_options = ["Married", "Single", "Divorced"]
        mar_val = get_val("MaritalStatus", "Married")
        marital = c3.selectbox("Marital status", mar_options, index=mar_options.index(mar_val) if mar_val in mar_options else 0)
        
        worklife = c3.slider("Work-life balance", 1, 4, int(get_val("WorkLifeBalance", 3)))
        involvement = c3.slider("Job involvement", 1, 4, int(get_val("JobInvolvement", 3)))

        st.markdown(
            '<div style="display:flex;align-items:center;gap:10px;margin:18px 0 16px;">'
            '<span class="tag">Role</span><span style="font-weight:800;font-size:15px;">Job & Organization</span></div>',
            unsafe_allow_html=True,
        )
        c4, c5, c6 = st.columns(3)
        role_options = ["Sales Executive", "Research Scientist", "Laboratory Technician",
                        "Manufacturing Director", "Healthcare Representative", "Manager",
                        "Sales Representative", "Research Director", "Human Resources"]
        role_val = get_val("JobRole", "Sales Executive")
        role = c4.selectbox("Job role", role_options, index=role_options.index(role_val) if role_val in role_options else 0)

        dept_options = ["Research & Development", "Sales", "Human Resources"]
        dept_val = get_val("Department", "Research & Development")
        dept = c4.selectbox("Department", dept_options, index=dept_options.index(dept_val) if dept_val in dept_options else 0)

        job_level = c4.number_input("Job level", 1, 5, int(get_val("JobLevel", 2)))

        travel_options = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
        travel_val = get_val("BusinessTravel", "Travel_Rarely")
        travel = c5.selectbox("Business travel", travel_options, index=travel_options.index(travel_val) if travel_val in travel_options else 0)

        gender_options = ["Male", "Female"]
        gender_val = get_val("Gender", "Male")
        gender = c5.selectbox("Gender", gender_options, index=gender_options.index(gender_val) if gender_val in gender_options else 0)

        edu_field_options = ["Life Sciences", "Medical", "Marketing",
                            "Technical Degree", "Human Resources", "Other"]
        edu_val = get_val("EducationField", "Life Sciences")
        edu_field = c5.selectbox("Education field", edu_field_options, index=edu_field_options.index(edu_val) if edu_val in edu_field_options else 0)

        education = c6.number_input("Education level", 1, 5, int(get_val("Education", 3)))
        stock = c6.number_input("Stock option level", 0, 3, int(get_val("StockOptionLevel", 1)))
        performance = c6.number_input("Performance rating", 1, 4, int(get_val("PerformanceRating", 3)))

        st.markdown(
            '<div style="display:flex;align-items:center;gap:10px;margin:18px 0 16px;">'
            '<span class="tag">Career</span><span style="font-weight:800;font-size:15px;">Tenure & Experience</span></div>',
            unsafe_allow_html=True,
        )
        c7, c8, c9 = st.columns(3)
        years_co = c7.number_input("Years at company", 0, 40, int(get_val("YearsAtCompany", 5)))
        total_work = c7.number_input("Total working years", 0, 40, int(get_val("TotalWorkingYears", 10)))
        years_role = c7.number_input("Years in current role", 0, 18, int(get_val("YearsInCurrentRole", 3)))
        years_mgr = c8.number_input("Years with current manager", 0, 17, int(get_val("YearsWithCurrManager", 3)))
        num_cos = c8.number_input("Companies worked", 0, 9, int(get_val("NumCompaniesWorked", 2)))
        last_promo = c8.number_input("Years since last promotion", 0, 15, int(get_val("YearsSinceLastPromotion", 2)))
        salary_hike = c9.number_input("Percent salary hike", 10, 25, int(get_val("PercentSalaryHike", 14)))
        training = c9.number_input("Training last year", 0, 6, int(get_val("TrainingTimesLastYear", 3)))
        rel_sat = c9.number_input("Relationship satisfaction", 1, 4, int(get_val("RelationshipSatisfaction", 3)))

        submitted = st.form_submit_button("Run Risk Assessment", use_container_width=True)

    sample = {
        "Age": age, "BusinessTravel": travel, "DailyRate": int(get_val("DailyRate", 800)), "Department": dept,
        "DistanceFromHome": distance, "Education": education, "EducationField": edu_field,
        "EnvironmentSatisfaction": env_sat, "Gender": gender, "HourlyRate": int(get_val("HourlyRate", 67)),
        "JobInvolvement": involvement, "JobLevel": job_level, "JobRole": role,
        "JobSatisfaction": satisfaction, "MaritalStatus": marital, "MonthlyIncome": income,
        "MonthlyRate": int(get_val("MonthlyRate", 12000)), "NumCompaniesWorked": num_cos, "OverTime": overtime,
        "PercentSalaryHike": salary_hike, "PerformanceRating": performance,
        "RelationshipSatisfaction": rel_sat, "StockOptionLevel": stock,
        "TotalWorkingYears": total_work, "TrainingTimesLastYear": training,
        "WorkLifeBalance": worklife, "YearsAtCompany": years_co,
        "YearsInCurrentRole": years_role, "YearsSinceLastPromotion": last_promo,
        "YearsWithCurrManager": years_mgr,
    }
    return submitted, sample


def page_predict():
    st.markdown(
        '<div style="font-size:26px;font-weight:800;color:#0F172A;letter-spacing:-.01em;">Risk Assessment</div>'
        '<div style="font-size:14px;color:#64748B;margin:2px 0 18px;">Evaluate individual employees or upload '
        'a batch workforce CSV for explainable attrition analytics.</div>',
        unsafe_allow_html=True,
    )
    b = get_bundle()

    tab_single, tab_batch = st.tabs(["👤 Single Employee Assessment", "📁 Batch Workforce Assessment (CSV)"])

    with tab_single:
        col_preset, _ = st.columns([1.5, 1])
        with col_preset:
            selected_preset_key = st.selectbox(
                "Quick-fill profile preset:",
                list(PRESETS.keys()),
                index=0,
                help="Pick a pre-configured employee persona to test risk scenarios instantly.",
            )
        preset_data = PRESETS.get(selected_preset_key)

        submitted, sample = prediction_form(preset_data)

        if submitted:
            with st.spinner("Computing risk..."):
                probability = predict_probability(b, sample)
            st.session_state["prob"] = probability
            st.session_state["sample"] = sample
            st.session_state["explained"] = False

        prob = st.session_state.get("prob")
        if prob is not None:
            level = risk_level(prob)
            c = RISK[level]
            verdict = prob >= b["threshold"]

            st.markdown('<div class="section-label">Assessment Result</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="panel" style="border-top:5px solid {c};">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap;">
                    <div style="flex:1;min-width:240px;">
                      <div style="display:flex;justify-content:space-between;align-items:baseline;">
                        <span style="font-size:13px;font-weight:600;color:#64748B;">Probability of attrition</span>
                        <span style="font-size:40px;font-weight:900;color:{c};letter-spacing:-.02em;">{prob:.1%}</span>
                      </div>
                      {seg_gauge(prob, c)}
                      <div style="display:flex;align-items:center;gap:10px;margin-top:16px;flex-wrap:wrap;">
                        {risk_pill(level)}
                        <span style="font-size:14px;font-weight:700;color:{'#7F1D1D' if verdict else '#046C4E'};">
                          {'Intervention recommended' if verdict else 'Monitor normally'}
                        </span>
                      </div>
                    </div>
                    <div style="text-align:right;">
                      <div style="font-size:11px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:#94A3B8;">verdict</div>
                      <div style="font-size:22px;font-weight:900;color:{'#7F1D1D' if verdict else '#046C4E'};margin-top:4px;">{'HIGH RISK' if verdict else 'LOW RISK'}</div>
                      <div style="font-size:11.5px;color:#94A3B8;margin-top:6px;">SVM · threshold {b['threshold']:.2f}</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            explain = st.button("Explain prediction · SHAP values", use_container_width=True)
            if explain or st.session_state.get("explained"):
                if not st.session_state.get("explained"):
                    with st.spinner("Decomposing the prediction into feature contributions..."):
                        st.session_state["factors"] = explain_prediction(b, st.session_state["sample"], top_n=6, n_samples=100)
                    st.session_state["explained"] = True

                factors, _ = st.session_state["factors"]
                panel_open("Why this verdict?", "red pushes toward attrition · green toward retention")
                fdf = pd.DataFrame(factors, columns=["Factor", "SHAP Contribution"])
                mx = max(abs(v) for _, v in factors)
                names = [f[0] for f in factors]
                vals = [f[1] for f in factors]
                fig = go.Figure(
                    go.Bar(
                        y=names, x=vals, orientation="h",
                        marker=dict(color=[RED if v >= 0 else EMERALD for v in vals], cornerradius=5),
                        text=[f"{v:+.3f}" for v in vals], textposition="outside",
                        textfont=dict(size=11, color=INK),
                        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
                    )
                )
                fig.update_xaxes(range=[-mx * 1.35, mx * 1.35])
                fig.update_layout(height=300)
                st.plotly_chart(style_fig(fig), use_container_width=True, config={"displayModeBar": False})
                st.dataframe(fdf.round(4), width="stretch", hide_index=True)
                panel_close()
        else:
            st.info("Select a preset or customize employee details above, then click 'Run Risk Assessment'.")

    with tab_batch:
        panel_open("Workforce Batch Scoring", "Upload an HR roster CSV or load sample records to score the entire department at once.")
        up_col1, up_col2 = st.columns([1.5, 1])
        with up_col1:
            uploaded_file = st.file_uploader("Upload Employee CSV", type=["csv"], help="Upload CSV containing employee demographic and HR features.")
        with up_col2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            use_sample = st.button("Load 100 Sample Employees (IBM HR)", use_container_width=True)

        batch_df = None
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(batch_df)} records from uploaded file.")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
        elif use_sample:
            raw_full = get_data()
            batch_df = raw_full.sample(n=min(100, len(raw_full)), random_state=42).reset_index(drop=True)
            st.session_state["sample_batch_df"] = batch_df

        if batch_df is None and "sample_batch_df" in st.session_state:
            batch_df = st.session_state["sample_batch_df"]

        if batch_df is not None:
            with st.spinner("Scoring workforce batch..."):
                scored_df = batch_predict(b, batch_df)

            total_emp = len(scored_df)
            n_churn = (scored_df["Attrition_Predicted"] == "Yes").sum()
            churn_pct = n_churn / total_emp if total_emp else 0
            n_crit_high = scored_df["Risk_Level"].isin(["High", "Critical"]).sum()

            b_c = st.columns(4)
            b_c[0].markdown(kpi("Assessed", f"{total_emp:,}", "total employees", "👥", PRIMARY, "#E8F0FE"), unsafe_allow_html=True)
            b_c[1].markdown(kpi("Predicted Leavers", f"{n_churn}", f"{churn_pct:.1%} of batch", "⚠️", RED, "#FEE2E2"), unsafe_allow_html=True)
            b_c[2].markdown(kpi("High / Critical Risk", f"{n_crit_high}", "immediate action recommended", "🔥", AMBER, "#FEF3C7"), unsafe_allow_html=True)
            b_c[3].markdown(kpi("Average Probability", f"{scored_df['Attrition_Probability'].mean():.1%}", "mean churn probability", "📊", VIOLET, "#F3E8FF"), unsafe_allow_html=True)

            ch_col1, ch_col2 = st.columns(2)
            with ch_col1:
                panel_open("Risk Tier Breakdown")
                risk_counts = scored_df["Risk_Level"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0)
                fig_risk = go.Figure(
                    go.Pie(
                        labels=risk_counts.index,
                        values=risk_counts.values,
                        hole=0.6,
                        marker=dict(colors=[EMERALD, AMBER, RED, "#7F1D1D"]),
                        textinfo="label+value",
                    )
                )
                st.plotly_chart(style_fig(fig_risk, 280), use_container_width=True, config={"displayModeBar": False})
                panel_close()

            with ch_col2:
                panel_open("Predicted Risk by Department")
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
                    st.plotly_chart(style_fig(fig_dep, 280), use_container_width=True, config={"displayModeBar": False})
                panel_close()

            st.markdown('<div class="section-label">High-Risk Employees Ranked</div>', unsafe_allow_html=True)
            display_cols = [c for c in ["EmployeeNumber", "JobRole", "Department", "MonthlyIncome", "OverTime", "WorkLifeBalance", "Attrition_Probability", "Risk_Level", "Attrition_Predicted"] if c in scored_df.columns]
            ranked_df = scored_df.sort_values("Attrition_Probability", ascending=False)
            st.dataframe(ranked_df[display_cols].head(25), width="stretch", hide_index=True)

            csv_buffer = io.StringIO()
            scored_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Full Scored Roster (CSV)",
                data=csv_buffer.getvalue(),
                file_name="employee_attrition_scored.csv",
                mime="text/csv",
                use_container_width=True,
            )
        panel_close()


# ---------------------------------------------------------------------------
def page_models():
    b = get_bundle()
    r = get_results()

    st.markdown(
        '<div style="font-size:26px;font-weight:800;color:#0F172A;letter-spacing:-.01em;">Model Performance</div>'
        '<div style="font-size:14px;color:#64748B;margin:2px 0 18px;">Five classifiers benchmarked against the '
        'same stratified 70/30 hold-out split.</div>',
        unsafe_allow_html=True,
    )
    fm = r["final_metrics"]
    c = st.columns(6)
    c[0].markdown(kpi("Champion", short_model(r["final_model"]), "tuned via search", "🏆", VIOLET, "#F3E8FF"), unsafe_allow_html=True)
    c[1].markdown(kpi("Accuracy", f"{fm['accuracy']:.2%}", "overall correct", "🎯", PRIMARY, "#E8F0FE"), unsafe_allow_html=True)
    c[2].markdown(kpi("Precision", f"{fm['precision']:.2%}", "leaver claims true", "🎯", PRIMARY, "#E8F0FE"), unsafe_allow_html=True)
    c[3].markdown(kpi("Recall", f"{fm['recall']:.2%}", "leavers caught", "🎯", PRIMARY, "#E8F0FE"), unsafe_allow_html=True)
    c[4].markdown(kpi("F1 Score", f"{fm['f1']:.2%}", "selection metric", "⭐", AMBER, "#FEF3C7"), unsafe_allow_html=True)
    c[5].markdown(kpi("ROC-AUC", f"{fm['roc_auc']:.3f}", "threshold-free", "🧭", EMERALD, "#D1FAE5"), unsafe_allow_html=True)

    st.markdown('<div class="section-label">Head to Head</div>', unsafe_allow_html=True)
    panel_open("F1 score by model", "SVM wins the benchmark; KNN shows why accuracy is not enough")
    rows = []
    for name, m in r["all_models"].items():
        rows.append({"Model": name, "accuracy": m["accuracy"], "precision": m["precision"],
                     "recall": m["recall"], "f1": m["f1"], "roc_auc": m["roc_auc"], "cv_f1": m["cv_f1_mean"]})
    table = pd.DataFrame(rows).sort_values("f1", ascending=False)
    colors = [PRIMARY if n == r["final_model"] else "#CBD5E1" for n in table["Model"]]
    fig = go.Figure(
        go.Bar(
            y=[short_model(n) for n in table["Model"]], x=table["f1"], orientation="h",
            marker=dict(color=colors, cornerradius=6),
            text=[f"{v:.3f}" for v in table["f1"]], textposition="outside",
            textfont=dict(size=12, color=INK),
            hovertemplate="%{y}: F1 %{x:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, table["f1"].max() * 1.2])
    st.plotly_chart(style_fig(fig, 330), use_container_width=True, config={"displayModeBar": False})
    panel_close()

    tab_roc, tab_cm, tab_sc = st.tabs(["ROC Curves", "Confusion Matrix", "F1 vs ROC-AUC"])
    with tab_roc:
        fig = go.Figure()
        for name, d in r["roc_data"].items():
            fig.add_trace(go.Scatter(x=d["fpr"], y=d["tpr"], mode="lines",
                                     name=f'{short_model(name)} · AUC {r["all_models"][name]["roc_auc"]:.3f}',
                                     line=dict(width=2.4 if name == r["final_model"] else 1.6,
                                               color=PRIMARY if name == r["final_model"] else "#94A3B8")))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="random",
                                 line=dict(color="#CBD5E1", dash="dash", width=1.2)))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate",
                          legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0))
        st.plotly_chart(style_fig(fig, 440), use_container_width=True, config={"displayModeBar": False})

    with tab_cm:
        cm = r["confusion_data"][r["final_model"]]
        z = np.array(cm)
        fig = go.Figure(go.Heatmap(
            z=z, x=["Predicted Stay", "Predicted Leave"], y=["Actual Stay", "Actual Leave"],
            colorscale="Blues", showscale=False,
            text=[[f"{int(v):,}" for v in row] for row in z], texttemplate="%{text}",
            textfont=dict(size=17, color="#fff", family="Inter"),
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text} employees<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="Prediction",
            yaxis_title="Ground Truth",
            yaxis=dict(autorange="reversed"),
        )
        fig.update_xaxes(side="bottom")
        tn, fp, fn, tp = z.ravel().tolist()
        st.markdown(
            f'<div style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap;">'
            f'<span class="tag">True Negatives (TN): {tn}</span><span class="tag">False Positives (FP): {fp}</span>'
            f'<span class="tag">False Negatives (FN): {fn}</span><span class="tag">True Positives (TP): {tp}</span></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(style_fig(fig, 360), use_container_width=True, config={"displayModeBar": False})

    with tab_sc:
        fig = go.Figure()
        for _, row in table.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["f1"]], y=[row["roc_auc"]], mode="markers+text",
                name=short_model(row["Model"]), text=[short_model(row["Model"])], textposition="top center",
                marker=dict(size=15, color=PRIMARY if row["Model"] == r["final_model"] else "#CBD5E1",
                            line=dict(color="#fff", width=1.5)),
                hovertemplate="%{text}: F1 %{x:.3f} · AUC %{y:.3f}<extra></extra>",
            ))
        fig.update_layout(xaxis_title="F1 score", yaxis_title="ROC-AUC",
                          showlegend=False, xaxis=dict(range=[0.2, 0.65]), yaxis=dict(range=[0.55, 0.9]))
        st.plotly_chart(style_fig(fig, 400), use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-label">Full Metrics</div>', unsafe_allow_html=True)
    panel_open("Performance table", f"decision threshold {fm['best_threshold']:.3f} · importance via {fm.get('importance_source', 'n/a')}")
    disp = table.copy()
    disp.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV F1"]
    disp.insert(0, "#", range(1, len(disp) + 1))
    st.dataframe(disp.round(3), width="stretch", hide_index=True)
    panel_close()


# ---------------------------------------------------------------------------
def page_explain():
    b = get_bundle()

    st.markdown(
        '<div style="font-size:26px;font-weight:800;color:#0F172A;letter-spacing:-.01em;">Explainability</div>'
        '<div style="font-size:14px;color:#64748B;margin:2px 0 18px;">Two complementary layers: what drives '
        'attrition globally, and why a single prediction looks the way it does.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.4, 1])
    with c1:
        panel_open("Global drivers — permutation importance", "drop in F1 when a feature is shuffled")
        top = pd.DataFrame(b["top_importance"])[::-1]
        fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"], orientation="h",
                               marker=dict(color=[VIOLET] * len(top), cornerradius=5),
                               hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        st.plotly_chart(style_fig(fig, 600), use_container_width=True, config={"displayModeBar": False})
        panel_close()

    with c2:
        panel_open("Top 15 drivers", "one-hot categories aggregated to base features")
        st.dataframe(top[["feature", "importance"]].round(4), width="stretch", hide_index=True)
        panel_close()

        panel_open("How to read SHAP", "local attribution per prediction")
        st.markdown(
            f"""
            For one employee profile, SHAP splits the model's probability into per-feature contributions:
            <br><br><div class="formula">P(attrition) = baseline + Σ feature contributions</div>
            <br>Positive contributions push toward <span style="color:{RED};font-weight:700;">attrition</span>,
            negative toward <span style="color:{EMERALD};font-weight:700;">retention</span>. The Risk
            Assessment page applies this per prediction and surfaces the biggest contributors.
            """,
            unsafe_allow_html=True,
        )
        panel_close()

        b_ = b
        panel_open("Champion model", "why SVM won")
        st.markdown(
            f"""
            <div style="font-size:13.5px;color:#334155;line-height:1.7;">
            <b style="color:#0F172A;">{short_model(b_['model_name'])}</b> balances precision
            ({b_['metrics']['precision']:.2f}) and recall ({b_['metrics']['recall']:.2f}) better than any
            alternative. As an RBF SVM it has no built-in tree importance, so explainability comes from
            <b>permutation importance</b> (global) and <b>SHAP</b> (local). The decision boundary was tuned
            with class-weight balancing for the 84/16 class mix.
            </div>
            """,
            unsafe_allow_html=True,
        )
        panel_close()


# ---------------------------------------------------------------------------
page = sidebar()
if page == "Dashboard":
    page_dashboard()
elif page == "Predict":
    page_predict()
elif page == "Models":
    page_models()
else:
    page_explain()