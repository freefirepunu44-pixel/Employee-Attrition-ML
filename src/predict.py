import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import CATEGORICAL_FEATURES, FEATURES, MODELS_PATH, NUMERICAL_FEATURES

_EXPLAINER_CACHE = {}


def load_model(bundle_path=MODELS_PATH / "best_model.pkl"):
    with open(bundle_path, "rb") as f:
        return joblib.load(f)


def _to_dataframe(sample: dict) -> pd.DataFrame:
    for col in FEATURES:
        if col not in sample:
            sample[col] = None
    return pd.DataFrame([sample])[FEATURES]


def risk_level(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    if probability < 0.55:
        return "Medium"
    if probability < 0.80:
        return "High"
    return "Critical"


def predict_probability(bundle, sample_dict: dict) -> float:
    df = _to_dataframe(dict(sample_dict))
    probabilities = bundle["pipeline"].predict_proba(df)[:, 1]
    return float(probabilities[0])


def batch_predict(bundle, df: pd.DataFrame) -> pd.DataFrame:
    """Predict attrition probability, binary outcome, and risk tier for a batch of employees."""
    df_clean = df.copy()
    for col in FEATURES:
        if col not in df_clean.columns:
            df_clean[col] = None
    X = df_clean[FEATURES]
    probs = bundle["pipeline"].predict_proba(X)[:, 1]
    threshold = float(bundle["threshold"])
    preds = (probs >= threshold).astype(int)
    risk_tiers = [risk_level(p) for p in probs]

    results = df.copy()
    results["Attrition_Probability"] = np.round(probs, 4)
    results["Attrition_Predicted"] = np.where(preds == 1, "Yes", "No")
    results["Risk_Level"] = risk_tiers
    return results


def _explainer(bundle, X_background=None):
    estimator = bundle["estimator"]
    cache_key = id(estimator)
    if cache_key in _EXPLAINER_CACHE:
        return _EXPLAINER_CACHE[cache_key]

    if hasattr(estimator, "tree_") or hasattr(estimator, "estimators_"):
        explainer = shap.TreeExplainer(estimator)
    else:
        background = np.asarray(bundle.get("background", X_background))
        if len(background) > 10:
            background = shap.kmeans(background, 10)
        explainer = shap.KernelExplainer(estimator.predict_proba, background)

    _EXPLAINER_CACHE[cache_key] = explainer
    return explainer


def explain_prediction(bundle, sample_dict: dict, top_n=5, n_samples=100):
    df = _to_dataframe(dict(sample_dict))
    preprocessor = bundle["preprocessor"]
    X_sample = preprocessor.transform(df)
    feature_names = bundle["feature_names"]

    explainer = _explainer(bundle, X_sample)

    if isinstance(explainer, shap.KernelExplainer):
        shap_nsamples = 100 if n_samples == "auto" else n_samples
        shap_values = explainer.shap_values(X_sample, nsamples=shap_nsamples, silent=True)
    else:
        shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[:, :, 1]

    encoded_to_base = bundle["encoded_to_base"]
    contributions = {}
    row = values[0] if values.ndim == 2 else values
    for col, value in zip(feature_names, row):
        base = encoded_to_base.get(col, col)
        contributions[base] = contributions.get(base, 0.0) + float(value)

    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return ranked[:top_n], contributions


def predict_with_explanation(bundle, sample_dict: dict):
    probability = predict_probability(bundle, sample_dict)
    threshold = bundle["threshold"]
    predicted_class = int(probability >= threshold)
    level = risk_level(probability)
    top_factors, contributions = explain_prediction(bundle, sample_dict)
    return {
        "probability": probability,
        "predicted_class": predicted_class,
        "risk_level": level,
        "threshold": threshold,
        "top_factors": top_factors,
        "contributions": contributions,
    }


if __name__ == "__main__":
    bundle = load_model()
    print(f"Model: {bundle['model_name']}")
    sample = {
        "Age": 28,
        "BusinessTravel": "Travel_Frequently",
        "DailyRate": 800,
        "Department": "Sales",
        "DistanceFromHome": 25,
        "Education": 2,
        "EducationField": "Marketing",
        "EnvironmentSatisfaction": 2,
        "Gender": "Male",
        "HourlyRate": 60,
        "JobInvolvement": 2,
        "JobLevel": 1,
        "JobRole": "Sales Executive",
        "JobSatisfaction": 2,
        "MaritalStatus": "Single",
        "MonthlyIncome": 25000,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 1,
        "OverTime": "Yes",
        "PercentSalaryHike": 11,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 3,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 2,
        "YearsAtCompany": 3,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 2,
    }
    result = predict_with_explanation(bundle, sample)
    print(f"Attrition probability: {result['probability']:.4f}")
    print(f"Risk level: {result['risk_level']}")
    print(f"Top factors pushing toward attrition: {result['top_factors']}")