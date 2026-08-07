# VisaPredict — AI-Based Visa Approval Prediction System

An end-to-end machine learning pipeline that predicts U.S. work-visa (OFLC) approval
outcomes from applicant, employer, and job attributes — cleaning raw application data,
training and comparing several scikit-learn classifiers, and automating prediction on
new applications.

## Results at a glance

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 70.3% | 0.813 | 0.673 |
| Decision Tree | 74.1% | 0.821 | 0.760 |
| Random Forest | 75.3% | 0.828 | 0.789 |
| **Gradient Boosting (best)** | **75.5%** | **0.828** | **0.786** |

- **Data cleaning:** cut the row-level data-quality issue rate (duplicates + missing
  fields) from ~23% to under 1% across 27,500+ ingested records
- **Automation:** the full pipeline — clean → engineer features → train 4 models →
  evaluate → save artifacts — runs end-to-end in ~11 seconds via one command
  (`src/run_pipeline.py`), replacing a manual, multi-step review process
- **Top predictors:** education level, wage unit, job experience, and continent of
  origin are the strongest drivers of approval

## Dataset

[EasyVisa](https://github.com/sandesha21/EasyVisa-Advanced-ML) — 25,480 real OFLC
(Office of Foreign Labor Certification) visa application records, each with:

| Field | Description |
|---|---|
| `continent` | Continent of the employee's origin |
| `education_of_employee` | Highest education level |
| `has_job_experience` | Y/N |
| `requires_job_training` | Y/N |
| `no_of_employees` | Employer's company size |
| `yr_of_estab` | Year the employer company was established |
| `region_of_employment` | US region of the job |
| `prevailing_wage` | Offered wage |
| `unit_of_wage` | Hour / Week / Month / Year |
| `full_time_position` | Y/N |
| `case_status` | Target: `Certified` (approved) or `Denied` |

**Note on data quality:** the public source dataset is already clean, so realistic
duplicate rows and missing values were deliberately injected
(`src/simulate_raw_data.py`) to build and demonstrate a genuine cleaning step — the
same kind of messiness you'd see exporting from a real case-management system. This is
documented here so it's easy to explain in an interview.

## Project structure

```
VisaPredict/
├── data/
│   ├── EasyVisa.csv              # original clean source (25,480 rows)
│   ├── EasyVisa_raw.csv          # simulated "as-ingested" data with noise
│   └── EasyVisa_clean.csv        # output of the cleaning step
├── notebooks/
│   └── VisaPredict_Analysis.ipynb  # full walkthrough: EDA -> cleaning -> modeling
├── src/
│   ├── simulate_raw_data.py      # injects realistic duplicates/missing values
│   ├── data_cleaning.py          # cleaning & preprocessing
│   ├── train.py                  # feature engineering + model training/comparison
│   ├── predict.py                # VisaPredictor class + CLI batch scoring
│   ├── eda.py                    # generates EDA plots
│   └── run_pipeline.py           # orchestrates the full automated workflow
├── models/                       # saved model, scaler, encoders (after training)
├── reports/
│   ├── training_summary.json     # metrics for every model
│   └── figures/                  # EDA + feature importance plots
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# Run the full automated pipeline (clean data -> train -> evaluate)
python src/run_pipeline.py

# Generate EDA plots
python src/eda.py

# Score a single application (as a Python library)
python -c "
from predict import VisaPredictor
p = VisaPredictor()
print(p.predict({
    'continent': 'Asia',
    'education_of_employee': \"Master's\",
    'has_job_experience': 'Y',
    'requires_job_training': 'N',
    'no_of_employees': 2500,
    'yr_of_estab': 2005,
    'region_of_employment': 'Northeast',
    'prevailing_wage': 85000,
    'unit_of_wage': 'Year',
    'full_time_position': 'Y',
}))
"

# Score a CSV of new applications
python src/predict.py --input new_applications.csv --output scored.csv
```

Or open `notebooks/VisaPredict_Analysis.ipynb` for the full interactive walkthrough
with charts.

## Approach

1. **Data cleaning** (`data_cleaning.py`) — standardizes text fields, removes exact
   duplicate rows, imputes missing categoricals with the mode and missing numerics
   with the median, and fixes impossible values (e.g. negative employee counts).
2. **Feature engineering** (`train.py`) — derives `company_age` from establishment
   year, label-encodes categorical fields, and standardizes numeric features.
3. **Modeling** — trains and compares Logistic Regression, Decision Tree, Random
   Forest, and Gradient Boosting classifiers on an 80/20 stratified train/test split;
   selects the best model by F1-score.
4. **Evaluation** — accuracy, F1, ROC-AUC, full classification report, and feature
   importance for the winning model.
5. **Automation** (`run_pipeline.py`, `predict.py`) — the entire workflow runs as one
   command, and new applications can be scored individually or in batch through a
   reusable `VisaPredictor` class.

## Tech stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Joblib · Jupyter

## Talking points for interviews / resume

- *"Built a machine learning pipeline using Python, Pandas, and Scikit-learn to
  analyze 25,000+ visa application records and predict approval outcomes"* —
  `train.py` trains and compares 4 classifiers on the full EasyVisa dataset.
- *"Reduced missing and duplicate data through systematic cleaning"* —
  `data_cleaning.py` cuts the data-quality issue rate from ~23% to <1%; the exact
  before/after numbers are in the notebook and `reports/training_summary.json`.
- *"Automated the end-to-end prediction workflow"* — `run_pipeline.py` chains
  cleaning → training → evaluation into a single reproducible command, and
  `predict.py` turns a trained model into a reusable scoring function/CLI.
