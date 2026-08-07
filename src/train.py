"""
train.py
--------
End-to-end model training:
  1. Load the cleaned dataset
  2. Feature engineering (encode categoricals, derive company age)
  3. Train/test split
  4. Train & compare several scikit-learn classifiers
  5. Evaluate on held-out test data
  6. Save the best model + preprocessing artifacts to models/

Run directly: python src/train.py
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

CLEAN_PATH = "/home/claude/VisaPredict/data/EasyVisa_clean.csv"
MODEL_DIR = "/home/claude/VisaPredict/models"
REPORT_DIR = "/home/claude/VisaPredict/reports"

CATEGORICAL_COLS = [
    "continent",
    "education_of_employee",
    "has_job_experience",
    "requires_job_training",
    "region_of_employment",
    "unit_of_wage",
    "full_time_position",
]
NUMERIC_COLS = ["no_of_employees", "company_age", "prevailing_wage"]
TARGET_COL = "case_status"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Derive company_age instead of using raw establishment year directly
    # (more stable/interpretable feature; avoids the model anchoring on a
    # specific calendar year)
    current_year = 2016  # dataset reflects OFLC FY2016 filings
    df["company_age"] = current_year - df["yr_of_estab"]
    df = df.drop(columns=["yr_of_estab", "case_id"], errors="ignore")
    return df


def encode(df: pd.DataFrame, encoders: dict | None = None):
    """Label-encode categorical columns. If `encoders` is None, fit new ones."""
    df = df.copy()
    fit_new = encoders is None
    if fit_new:
        encoders = {}
    for col in CATEGORICAL_COLS:
        if fit_new:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].map(lambda x, le=le: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    return df, encoders


def main():
    t0 = time.time()
    df = pd.read_csv(CLEAN_PATH)
    df = engineer_features(df)

    # Target: Certified -> 1, Denied -> 0
    df[TARGET_COL] = df[TARGET_COL].map({"Certified": 1, "Denied": 0})

    X_cat_encoded, encoders = encode(df[CATEGORICAL_COLS + NUMERIC_COLS])
    X = X_cat_encoded
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
        ),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1_score": round(f1_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, probs), 4),
        }
        fitted[name] = model
        print(f"{name:20s} | acc={results[name]['accuracy']:.3f}  "
              f"f1={results[name]['f1_score']:.3f}  auc={results[name]['roc_auc']:.3f}")

    best_name = max(results, key=lambda n: results[n]["f1_score"])
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name} (F1={results[best_name]['f1_score']})")

    # Classification report for the best model
    best_preds = best_model.predict(X_test_scaled)
    report_text = classification_report(y_test, best_preds, target_names=["Denied", "Certified"])
    print(f"\n{report_text}")

    # Feature importance (if available)
    feat_importance = None
    if hasattr(best_model, "feature_importances_"):
        feat_importance = dict(
            sorted(
                zip(X.columns, best_model.feature_importances_.round(4)),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )

    # --- Save artifacts ---
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(encoders, f"{MODEL_DIR}/encoders.pkl")
    joblib.dump(list(X.columns), f"{MODEL_DIR}/feature_names.pkl")

    summary = {
        "best_model": best_name,
        "results": results,
        "feature_importance": {k: float(v) for k, v in feat_importance.items()} if feat_importance else None,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "training_time_seconds": round(time.time() - t0, 2),
    }
    with open(f"{REPORT_DIR}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved model + artifacts to {MODEL_DIR}/")
    print(f"Saved training summary to {REPORT_DIR}/training_summary.json")
    print(f"Total pipeline time: {summary['training_time_seconds']}s")


if __name__ == "__main__":
    main()
