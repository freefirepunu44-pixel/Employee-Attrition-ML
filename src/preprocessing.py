from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "employee_attrition.csv"
MODELS_PATH = ROOT / "models"
OUTPUTS_PATH = ROOT / "outputs"
PLOTS_PATH = OUTPUTS_PATH / "plots"

TARGET = "Attrition"
DROP_COLUMNS = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]

NUMERICAL_FEATURES = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

CATEGORICAL_FEATURES = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_dataframe(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = df[TARGET].map({"No": 0, "Yes": 1})
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def transformed_feature_names(preprocessor: ColumnTransformer):
    names = list(NUMERICAL_FEATURES)
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    for base_name, categories in zip(
        CATEGORICAL_FEATURES, encoder.categories_
    ):
        for cat in categories:
            names.append(f"{base_name}_{cat}")
    return names