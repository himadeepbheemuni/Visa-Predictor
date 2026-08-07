"""
eda.py
------
Generates exploratory data analysis plots for the clean visa dataset
and saves them to reports/figures/.

Run directly: python src/eda.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

CLEAN_PATH = "/home/claude/VisaPredict/data/EasyVisa_clean.csv"
FIG_DIR = "/home/claude/VisaPredict/reports/figures"


def main():
    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(CLEAN_PATH)

    # 1. Overall approval rate
    fig, ax = plt.subplots(figsize=(5, 4))
    df["case_status"].value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452"], ax=ax)
    ax.set_title("Visa Application Outcomes")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Applications")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/outcome_distribution.png", dpi=120)
    plt.close()

    # 2. Approval rate by education level
    edu_order = ["High School", "Bachelor's", "Master's", "Doctorate"]
    approval_by_edu = (
        df.groupby("education_of_employee")["case_status"]
        .apply(lambda s: (s == "Certified").mean())
        .reindex(edu_order)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    approval_by_edu.plot(kind="bar", color="#55A868", ax=ax)
    ax.set_title("Approval Rate by Education Level")
    ax.set_ylabel("Approval Rate")
    ax.set_xlabel("")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/approval_by_education.png", dpi=120)
    plt.close()

    # 3. Approval rate by region of employment
    approval_by_region = (
        df.groupby("region_of_employment")["case_status"]
        .apply(lambda s: (s == "Certified").mean())
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    approval_by_region.plot(kind="bar", color="#C44E52", ax=ax)
    ax.set_title("Approval Rate by Region of Employment")
    ax.set_ylabel("Approval Rate")
    ax.set_xlabel("")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/approval_by_region.png", dpi=120)
    plt.close()

    # 4. Prevailing wage distribution by outcome (yearly-normalized approx, raw for simplicity)
    fig, ax = plt.subplots(figsize=(6, 4))
    yearly = df[df["unit_of_wage"] == "Year"]
    sns.boxplot(
        data=yearly[yearly["prevailing_wage"] < yearly["prevailing_wage"].quantile(0.95)],
        x="case_status", y="prevailing_wage", ax=ax, palette=["#DD8452", "#4C72B0"],
    )
    ax.set_title("Prevailing Wage by Outcome (Yearly-Paid Roles)")
    ax.set_xlabel("")
    ax.set_ylabel("Prevailing Wage (USD)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/wage_by_outcome.png", dpi=120)
    plt.close()

    # 5. Feature importance from trained model (if available)
    import json
    try:
        with open("/home/claude/VisaPredict/reports/training_summary.json") as f:
            summary = json.load(f)
        if summary.get("feature_importance"):
            fi = pd.Series(summary["feature_importance"]).sort_values()
            fig, ax = plt.subplots(figsize=(6, 5))
            fi.plot(kind="barh", color="#8172B2", ax=ax)
            ax.set_title(f"Feature Importance ({summary['best_model']})")
            plt.tight_layout()
            plt.savefig(f"{FIG_DIR}/feature_importance.png", dpi=120)
            plt.close()
    except FileNotFoundError:
        pass

    print(f"Saved plots to {FIG_DIR}/")


if __name__ == "__main__":
    main()
