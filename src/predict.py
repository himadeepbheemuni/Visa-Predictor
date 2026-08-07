"""
predict.py
----------
Load the trained model and score new visa applications.

Usage as a library:
    from predict import VisaPredictor
    p = VisaPredictor()
    result = p.predict({
        "continent": "Asia",
        "education_of_employee": "Master's",
        "has_job_experience": "Y",
        "requires_job_training": "N",
        "no_of_employees": 2500,
        "yr_of_estab": 2005,
        "region_of_employment": "Northeast",
        "prevailing_wage": 85000,
        "unit_of_wage": "Year",
        "full_time_position": "Y",
    })

Usage from the command line (scores a CSV of new applications):
    python src/predict.py --input new_applications.csv --output scored.csv
"""
import argparse

import joblib
import pandas as pd

from train import CATEGORICAL_COLS, NUMERIC_COLS, engineer_features

MODEL_DIR = "/home/claude/VisaPredict/models"


class VisaPredictor:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model = joblib.load(f"{model_dir}/best_model.pkl")
        self.scaler = joblib.load(f"{model_dir}/scaler.pkl")
        self.encoders = joblib.load(f"{model_dir}/encoders.pkl")
        self.feature_names = joblib.load(f"{model_dir}/feature_names.pkl")

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        df = engineer_features(df)
        df = df[self.feature_names].copy()
        for col in CATEGORICAL_COLS:
            le = self.encoders[col]
            df[col] = df[col].map(lambda x, le=le: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
        return df

    def predict(self, application: dict) -> dict:
        """Score a single application (dict of raw field values)."""
        df = pd.DataFrame([application])
        X = self._prep(df)
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        return {
            "prediction": "Certified" if pred == 1 else "Denied",
            "approval_probability": round(float(proba[1]), 4),
            "confidence": round(float(max(proba)), 4),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a dataframe of applications; returns predictions appended."""
        X = self._prep(df)
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        out = df.copy()
        out["predicted_status"] = ["Certified" if p == 1 else "Denied" for p in preds]
        out["approval_probability"] = probs.round(4)
        return out


def main():
    parser = argparse.ArgumentParser(description="Score visa applications with the trained model.")
    parser.add_argument("--input", required=True, help="CSV of applications to score")
    parser.add_argument("--output", required=True, help="Where to write the scored CSV")
    args = parser.parse_args()

    predictor = VisaPredictor()
    df_in = pd.read_csv(args.input)
    df_out = predictor.predict_batch(df_in)
    df_out.to_csv(args.output, index=False)
    print(f"Scored {len(df_out)} applications -> {args.output}")
    print(df_out["predicted_status"].value_counts())


if __name__ == "__main__":
    main()
