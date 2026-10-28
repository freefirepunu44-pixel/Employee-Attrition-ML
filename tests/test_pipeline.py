import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate import compute_metrics, compute_threshold_curve
from predict import (
    batch_predict,
    explain_prediction,
    load_model,
    predict_probability,
    predict_with_explanation,
    risk_level,
)
from preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
    load_data,
    prepare_dataframe,
    transformed_feature_names,
)


class TestDataAndPreprocessing(unittest.TestCase):
    def setUp(self):
        self.df = load_data()

    def test_load_data_shape_and_columns(self):
        self.assertEqual(len(self.df), 1470)
        self.assertIn("Attrition", self.df.columns)
        self.assertEqual(self.df.isna().sum().sum(), 0)

    def test_prepare_dataframe(self):
        X, y = prepare_dataframe(self.df)
        self.assertEqual(len(X), len(self.df))
        self.assertEqual(list(X.columns), FEATURES)
        self.assertTrue(set(y.unique()).issubset({0, 1}))
        self.assertEqual(y.sum(), (self.df["Attrition"] == "Yes").sum())

    def test_preprocessor_transformation(self):
        X, y = prepare_dataframe(self.df)
        preprocessor = build_preprocessor()
        preprocessor.fit(X, y)
        X_trans = preprocessor.transform(X.head(20))
        names = transformed_feature_names(preprocessor)
        self.assertEqual(X_trans.shape[1], len(names))
        self.assertEqual(len(names), 51)


class TestPredictionService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_model()
        cls.sample = {
            "Age": 32,
            "BusinessTravel": "Travel_Frequently",
            "DailyRate": 700,
            "Department": "Sales",
            "DistanceFromHome": 15,
            "Education": 3,
            "EducationField": "Marketing",
            "EnvironmentSatisfaction": 2,
            "Gender": "Male",
            "HourlyRate": 60,
            "JobInvolvement": 2,
            "JobLevel": 1,
            "JobRole": "Sales Executive",
            "JobSatisfaction": 2,
            "MaritalStatus": "Single",
            "MonthlyIncome": 3500,
            "MonthlyRate": 12000,
            "NumCompaniesWorked": 2,
            "OverTime": "Yes",
            "PercentSalaryHike": 12,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 2,
            "StockOptionLevel": 0,
            "TotalWorkingYears": 5,
            "TrainingTimesLastYear": 2,
            "WorkLifeBalance": 2,
            "YearsAtCompany": 3,
            "YearsInCurrentRole": 2,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 2,
        }

    def test_load_model_bundle_keys(self):
        for key in ["pipeline", "preprocessor", "estimator", "model_name", "threshold", "feature_names"]:
            self.assertIn(key, self.bundle)

    def test_risk_level_mapping(self):
        self.assertEqual(risk_level(0.15), "Low")
        self.assertEqual(risk_level(0.40), "Medium")
        self.assertEqual(risk_level(0.65), "High")
        self.assertEqual(risk_level(0.85), "Critical")

    def test_predict_probability_bounds(self):
        prob = predict_probability(self.bundle, self.sample)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_predict_with_explanation(self):
        res = predict_with_explanation(self.bundle, self.sample)
        self.assertIn("probability", res)
        self.assertIn("predicted_class", res)
        self.assertIn("risk_level", res)
        self.assertIn("top_factors", res)
        self.assertIn("contributions", res)
        self.assertIn(res["predicted_class"], [0, 1])
        self.assertGreater(len(res["top_factors"]), 0)

    def test_batch_predict(self):
        df_sample = pd.DataFrame([self.sample, self.sample])
        scored = batch_predict(self.bundle, df_sample)
        self.assertEqual(len(scored), 2)
        self.assertIn("Attrition_Probability", scored.columns)
        self.assertIn("Attrition_Predicted", scored.columns)
        self.assertIn("Risk_Level", scored.columns)
        self.assertTrue(all(scored["Attrition_Probability"] >= 0))
        self.assertTrue(all(scored["Attrition_Probability"] <= 1))
        self.assertTrue(all(scored["Attrition_Predicted"].isin(["Yes", "No"])))


class TestEvaluationMetrics(unittest.TestCase):
    def test_compute_metrics(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.4, 0.3, 0.9])
        metrics = compute_metrics(y_true, y_pred, y_proba)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.8)

    def test_compute_threshold_curve(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.7, 0.6, 0.3, 0.85, 0.15, 0.75])
        best_th, best_f1 = compute_threshold_curve(y_true, y_proba)
        self.assertGreater(best_th, 0.0)
        self.assertLessEqual(best_th, 1.0)
        self.assertGreater(best_f1, 0.5)


if __name__ == "__main__":
    unittest.main()
