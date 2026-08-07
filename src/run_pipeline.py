"""
run_pipeline.py
----------------
Automates the entire VisaPredict workflow in one command:
  raw data -> cleaning -> feature engineering -> model training ->
  evaluation -> saved model artifacts -> summary report

This is the piece that replaces what used to be a manual, multi-step
process (opening the data in a spreadsheet, manually flagging bad rows,
retraining a model by hand, re-checking metrics) with a single reproducible
script. On this dataset it runs end-to-end in well under a minute --
the "60% faster than manual analysis" claim refers to this automation
replacing that manual review workflow.

Usage: python src/run_pipeline.py
"""
import time

import data_cleaning
import train


def main():
    print("=" * 60)
    print("VisaPredict — Automated Pipeline")
    print("=" * 60)

    t0 = time.time()

    print("\n[1/2] Cleaning data...")
    t1 = time.time()
    data_cleaning.main()
    clean_time = time.time() - t1

    print("\n[2/2] Training & evaluating models...")
    t2 = time.time()
    train.main()
    train_time = time.time() - t2

    total_time = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Pipeline complete in {total_time:.1f}s "
          f"(cleaning: {clean_time:.1f}s, training: {train_time:.1f}s)")
    print("Model + artifacts:  models/")
    print("Metrics summary:    reports/training_summary.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
