# Examples

`quickstart.py` runs a temporary Day 1 smoke reproduction from `configs/day1.yaml`: it ingests the reference table, trains one XGBoost iteration, evaluates the test split, and prints the binary-collapsed F1.

`v_sterol_ensemble.py` is a post-run analysis helper for the recommended `configs/v_sterol.yaml` ensemble. It loads persisted prediction parquet output, averages model probabilities, mines residual lipid confusions for true STE rows, prints the top errors, and writes `reports/v_sterol/ste_confusion_mining.csv`.
