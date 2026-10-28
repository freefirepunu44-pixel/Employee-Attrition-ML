from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "employee_attrition.csv"
EDA_OUT = ROOT / "outputs" / "plots" / "eda"

plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "font.family": "sans-serif"})
sns.set_theme(style="whitegrid")

NAVY = "#1F3A5F"
BLUE = "#2F6690"
ORANGE = "#F4A259"
RED = "#D90429"
GRAY = "#6B7280"


def generate_all_eda():
    EDA_OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape}")

    # 1. Target Distribution
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    counts = df["Attrition"].value_counts()
    ax[0].bar(counts.index, counts.values, color=[BLUE, ORANGE], edgecolor="black", linewidth=0.5)
    for bar in ax[0].patches:
        ax[0].annotate(f"{bar.get_height():,}\n({bar.get_height()/len(df):.1%})",
                       (bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                       ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    ax[0].set_title("Attrition Class Frequencies (Imbalance)", fontsize=12, fontweight="bold")
    ax[0].set_ylabel("Employee Count")

    ax[1].pie(counts.values, labels=["Retained (No)", "Departed (Yes)"], autopct="%1.1f%%",
              colors=[BLUE, ORANGE], startangle=90, explode=(0, 0.08),
              wedgeprops=dict(edgecolor="black", linewidth=0.6))
    ax[1].set_title("Attrition Proportions (83.9% / 16.1%)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p1 = EDA_OUT / "eda_target_distribution.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p1}")

    # 2. Overtime & Marital Status
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ot = df.groupby("OverTime")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)
    bars_ot = ax[0].bar(ot.index, ot.values, color=[BLUE, RED], edgecolor="black", linewidth=0.5)
    for bar in bars_ot:
        ax[0].annotate(f"{bar.get_height():.1f}%",
                       (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8),
                       ha="center", va="bottom", fontweight="bold")
    ax[0].set_title("Attrition Rate by OverTime Status", fontsize=12, fontweight="bold")
    ax[0].set_ylabel("Attrition Rate (%)")
    ax[0].set_ylim(0, max(ot.values) * 1.25)

    ms = df.groupby("MaritalStatus")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100).sort_values(ascending=False)
    bars_ms = ax[1].bar(ms.index, ms.values, color=NAVY, edgecolor="black", linewidth=0.5)
    for bar in bars_ms:
        ax[1].annotate(f"{bar.get_height():.1f}%",
                       (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8),
                       ha="center", va="bottom", fontweight="bold")
    ax[1].set_title("Attrition Rate by Marital Status", fontsize=12, fontweight="bold")
    ax[1].set_ylabel("Attrition Rate (%)")
    ax[1].set_ylim(0, max(ms.values) * 1.25)
    fig.tight_layout()
    p2 = EDA_OUT / "eda_overtime_marital.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p2}")

    # 3. Department & Job Role
    fig, ax = plt.subplots(figsize=(10, 5))
    role_rate = df.groupby("JobRole")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100).sort_values()
    bars = ax.barh(role_rate.index, role_rate.values, color=BLUE, edgecolor="black", linewidth=0.5)
    for bar in bars:
        ax.annotate(f"{bar.get_width():.1f}%",
                    (bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2),
                    va="center", ha="left", fontweight="bold", fontsize=9)
    ax.set_title("Attrition Rate Across Specific Job Roles", fontsize=12, fontweight="bold")
    ax.set_xlabel("Attrition Rate (%)")
    ax.set_xlim(0, max(role_rate.values) * 1.18)
    fig.tight_layout()
    p3 = EDA_OUT / "eda_jobrole_attrition.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p3}")

    # 4. Satisfaction Metrics (4-panel trends)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    sats = [
        ("JobSatisfaction", "Job Satisfaction"),
        ("EnvironmentSatisfaction", "Environment Sat."),
        ("WorkLifeBalance", "Work-Life Balance"),
        ("JobInvolvement", "Job Involvement"),
    ]
    for ax, (col, title) in zip(axes, sats):
        rate = df.groupby(col)["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)
        ax.plot(rate.index, rate.values, marker="o", linewidth=2.4, color=RED, markersize=7)
        for x, y in zip(rate.index, rate.values):
            ax.annotate(f"{y:.1f}%", (x, y + 1.2), ha="center", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Attrition Rate (%)" if ax == axes[0] else "")
        ax.set_xticks(sorted(df[col].unique()))
        ax.set_ylim(0, max(rate.values) * 1.35)
    fig.suptitle("Attrition Rates across Employee Satisfaction Dimensions (1=Low, 4=Very High)", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    p4 = EDA_OUT / "eda_satisfaction_trends.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p4}")

    # 5. Continuous Boxplots (Age, Income, Distance, Tenure)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    cont = [("Age", "Age (Years)"),
            ("MonthlyIncome", "Monthly Income ($)"),
            ("DistanceFromHome", "Distance (Miles)"),
            ("TotalWorkingYears", "Total Working Years")]
    for ax, (col, title) in zip(axes, cont):
        sns.boxplot(data=df, x="Attrition", y=col, ax=ax, palette=[BLUE, ORANGE],
                    boxprops=dict(edgecolor="black", linewidth=0.6))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Attrition Status")
    fig.suptitle("Distribution of Key Continuous Variables by Attrition Status", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    p5 = EDA_OUT / "eda_continuous_boxplots.png"
    fig.savefig(p5, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p5}")

    # 6. Correlation Heatmap
    df_num = df.copy()
    df_num["Attrition"] = (df_num["Attrition"] == "Yes").astype(int)
    numeric_cols = df_num.select_dtypes(include=[np.number]).columns
    corr = df_num[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=False,
                square=True, linewidths=0.4, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Heatmap: Numeric Features vs. Attrition", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p6 = EDA_OUT / "eda_correlation_heatmap.png"
    fig.savefig(p6, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p6}")

    print("All 6 EDA figures generated successfully.")


if __name__ == "__main__":
    generate_all_eda()
