import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import (
    compute_metrics,
    compute_threshold_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_roc_curves,
    save_metrics_json,
)
from preprocessing import (
    CATEGORICAL_FEATURES,
    MODELS_PATH,
    NUMERICAL_FEATURES,
    OUTPUTS_PATH,
    PLOTS_PATH,
    build_preprocessor,
    load_data,
    prepare_dataframe,
    transformed_feature_names,
)

RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 5

MODEL_DEFS = {
    "Logistic Regression": Pipeline(
        steps=[("pre", build_preprocessor()), ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE))]
    ),
    "Decision Tree": Pipeline(
        steps=[("pre", build_preprocessor()), ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE))]
    ),
    "Random Forest": Pipeline(
        steps=[("pre", build_preprocessor()), ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE))]
    ),
    "K-Nearest Neighbors": Pipeline(
        steps=[("pre", build_preprocessor()), ("clf", KNeighborsClassifier())]
    ),
    "Support Vector Machine": Pipeline(
        steps=[("pre", build_preprocessor()), ("clf", SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE))]
    ),
}

TUNING_GRIDS = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__solver": ["liblinear"],
    },
    "Decision Tree": {
        "clf__max_depth": [3, 5, 7, 10, None],
        "clf__min_samples_leaf": [1, 2, 5, 10],
    },
    "Random Forest": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [6, 8, 10, None],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2"],
    },
    "K-Nearest Neighbors": {
        "clf__n_neighbors": [3, 5, 7, 9, 11, 15],
        "clf__weights": ["uniform", "distance"],
    },
    "Support Vector Machine": {
        "clf__C": [0.1, 0.5, 1, 5, 10],
        "clf__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    },
}

RANDOMIZED_MODELS = {"Random Forest", "Support Vector Machine"}
RANDOMIZED_N_ITER = 25


def train_baseline(X_train, y_train, X_test, y_test):
    results = {}
    tuned_pipelines = {}
    for name, pipeline in MODEL_DEFS.items():
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
        cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_pred, y_proba)
        threshold, threshold_f1 = compute_threshold_curve(y_test, y_proba)
        test_metrics["cv_f1_mean"] = round(float(cv_f1.mean()), 4)
        test_metrics["cv_f1_std"] = round(float(cv_f1.std()), 4)
        test_metrics["cv_auc_mean"] = round(float(cv_auc.mean()), 4)
        test_metrics["best_threshold"] = threshold
        test_metrics["best_threshold_f1"] = round(threshold_f1, 4)

        results[name] = test_metrics
        tuned_pipelines[name] = pipeline
        print(f"[baseline] {name}: test-F1={test_metrics['f1']:.4f} "
              f"test-AUC={test_metrics['roc_auc']:.4f} "
              f"cv-F1={test_metrics['cv_f1_mean']:.4f} (run={time.strftime('%H:%M:%S')})")
    return results, tuned_pipelines


def tune_model(name, pipeline, X_train, y_train):
    param_grid = TUNING_GRIDS[name]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    if name in RANDOMIZED_MODELS:
        searcher = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=RANDOMIZED_N_ITER,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0,
        )
    else:
        searcher = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=0
        )
    searcher.fit(X_train, y_train)
    print(f"[tuned ] {name}: best_params={searcher.best_params_} "
          f"cv-F1={searcher.best_score_:.4f}")
    return searcher.best_estimator_, searcher.best_params_, searcher.best_score_


def main():
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X, y = prepare_dataframe(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    baseline_results, tuned_pipelines = train_baseline(X_train, y_train, X_test, y_test)

    roc_data = {}
    cm_data = {}

    for name in MODEL_DEFS:
        pipeline = tuned_pipelines[name]
        best_estimator, best_params, best_cv = tune_model(name, pipeline, X_train, y_train)
        tuned_pipelines[name] = best_estimator
        y_pred = best_estimator.predict(X_test)
        y_proba = best_estimator.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)
        threshold, threshold_f1 = compute_threshold_curve(y_test, y_proba)
        metrics["cv_f1_mean"] = round(float(best_cv), 4)
        metrics["best_threshold"] = threshold
        metrics["best_threshold_f1"] = round(threshold_f1, 4)
        metrics["best_params"] = {k.split("clf__")[-1]: v for k, v in best_params.items()}
        baseline_results[name] = metrics
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        cm = confusion_matrix(y_test, (y_proba >= threshold).astype(int))
        cm_data[name] = cm.tolist()

    results_df = pd.DataFrame(baseline_results).T
    results_df.index.name = "model"
    results_df = results_df.reset_index()

    final_rank = results_df.sort_values("f1", ascending=False).iloc[0]
    model_name = final_rank["model"]
    print(f"\nBest model by test F1: {model_name} (F1={final_rank['f1']:.4f})")

    final_pipeline = tuned_pipelines[model_name]
    preprocessor = final_pipeline.named_steps["pre"]
    estimator = final_pipeline.named_steps["clf"]
    feature_names = transformed_feature_names(preprocessor)

    y_proba_final = final_pipeline.predict_proba(X_test)[:, 1]
    best_threshold = float(baseline_results[model_name]["best_threshold"])
    y_pred_threshold = (y_proba_final >= best_threshold).astype(int)
    final_metrics = compute_metrics(y_test, y_pred_threshold, y_proba_final)
    final_metrics["best_threshold"] = best_threshold

    final_metrics["cv_f1_mean"] = float(final_rank["cv_f1_mean"])
    final_metrics["best_params"] = final_rank["best_params"]

    roc_path = plot_roc_curves(
        [(m, y_test, tuned_pipelines[m].predict_proba(X_test)[:, 1]) for m in MODEL_DEFS]
    )
    compare_path = plot_model_comparison(results_df, metric="f1")
    cm_path = plot_confusion_matrix(y_test, y_pred_threshold, model_name)

    feature_names = transformed_feature_names(preprocessor)
    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(estimator, "feature_importances_"):
        feature_importance = estimator.feature_importances_
        importance_source = "built-in"
    else:
        perm = permutation_importance(
            estimator, X_test_transformed, y_test, n_repeats=10, random_state=RANDOM_STATE
        )
        feature_importance = perm.importances_mean
        importance_source = "permutation"
    importance_path = plot_feature_importance(
        feature_importance, feature_names, title=f"Top Feature Importance ({importance_source.capitalize()})"
    )

    importance_df = pd.DataFrame({"feature": feature_names, "importance": feature_importance})
    importance_df = importance_df.sort_values("importance", ascending=False)
    top_importance = importance_df.head(15).to_dict("records")

    final_metrics["importance_source"] = importance_source

    encoded_to_base = {}
    for col in feature_names:
        base = col.split("_")[0] if "_" in col else col
        encoded_to_base[col] = base

    bundle = {
        "pipeline": final_pipeline,
        "preprocessor": preprocessor,
        "estimator": estimator,
        "model_name": model_name,
        "threshold": best_threshold,
        "feature_names": feature_names,
        "encoded_to_base": encoded_to_base,
        "background": preprocessor.transform(X_train)[:100],
        "top_importance": top_importance,
        "metrics": final_metrics,
        "all_model_metrics": baseline_results,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features_raw": int(X.shape[1]),
    }

    bundle_path = MODELS_PATH / "best_model.pkl"
    with open(bundle_path, "wb") as f:
        joblib.dump(bundle, f)

    metrics_json_path = OUTPUTS_PATH / "model_results.json"
    save_metrics_json(
        {
            "final_model": model_name,
            "final_metrics": final_metrics,
            "top_importance": top_importance,
            "all_models": baseline_results,
            "roc_data": roc_data,
            "confusion_data": cm_data,
        },
        metrics_json_path,
    )

    print("\nSaved artifacts:")
    for p in [bundle_path, metrics_json_path, roc_path, compare_path, cm_path, importance_path]:
        if p is not None:
            print(f"  {p}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()