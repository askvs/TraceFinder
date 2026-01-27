#!/usr/bin/env python3
"""
tune_xgb_gb_optuna.py

Tune XGBoost and sklearn GradientBoostingClassifier with Optuna.
Pipeline: StandardScaler -> ADASYN -> classifier
Objective: maximize cross-validated f1_macro.

Saves:
 - best pipelines: models/XGBoost_best.pkl, models/GradientBoosting_best.pkl
 - optuna study objects: models/study_xgb.pkl, models/study_gb.pkl
 - label encoder: models/label_encoder.pkl
 - results summary CSV: models/tuning_results_optuna.csv
"""

import os
import warnings
from time import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

# classifiers
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

import optuna
from optuna.samplers import TPESampler

# Config
CSV_PATH = "processed_data/merged_metadata.csv"
MODELS_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_TRIALS = 40   # reduce/increase depending on compute/time budget

os.makedirs(MODELS_DIR, exist_ok=True)
warnings.filterwarnings("ignore")  # optional: hide sklearn/xgboost warnings for clarity

# -------------------------
# Utility helpers
# -------------------------
def build_pipeline(clf):
    """Return an imblearn pipeline wrapping scaler + ADASYN + classifier."""
    return ImbPipeline([("scaler", StandardScaler()), ("adasyn", ADASYN(random_state=RANDOM_STATE)), ("clf", clf)])

def cv_score_pipeline(pipeline, X, y, cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)):
    """Return mean cross-validated f1_macro (using n_jobs=-1)."""
    scores = cross_val_score(pipeline, X, y, scoring="f1_macro", cv=cv, n_jobs=-1, error_score="raise")
    return float(np.mean(scores))

# -------------------------
# Load and prepare data
# -------------------------
df = pd.read_csv(CSV_PATH)
if "class_label" not in df.columns:
    raise KeyError("CSV must contain 'class_label' column.")

# features & labels
X = df.drop(columns=["file_name", "main_class", "resolution", "class_label"], errors="ignore")
y_raw = df["class_label"].astype(str).copy()

# encode labels to integers (necessary for XGBoost)
le = LabelEncoder()
y = le.fit_transform(y_raw)
joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

print("Classes:", list(le.classes_))
print("Dataset shape:", X.shape)
print("Class counts:")
print(pd.Series(y_raw).value_counts().sort_index())

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
print(f"\nTrain: {len(y_train)} samples, Test: {len(y_test)} samples")
print(pd.Series(le.inverse_transform(y_train)).value_counts().sort_index())

# CV splitter
cv_split = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# -------------------------
# Optuna objective: XGBoost
# -------------------------
def objective_xgb(trial):
    # Parameter search space (reasonable defaults)
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
        "use_label_encoder": False,
        "random_state": RANDOM_STATE,
        "verbosity": 0,
        "n_jobs": -1,
        "objective": "multi:softprob",
    }
    clf = XGBClassifier(**param)
    pipeline = build_pipeline(clf)
    try:
        score = cv_score_pipeline(pipeline, X_train, y_train, cv=cv_split)
    except Exception as e:
        # If pipeline fails for some params, return a poor score
        return 0.0
    return score

# -------------------------
# Optuna objective: GradientBoosting (sklearn)
# -------------------------
def objective_gb(trial):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2", None]),
        "random_state": RANDOM_STATE,
    }
    clf = GradientBoostingClassifier(**param)
    pipeline = build_pipeline(clf)
    try:
        score = cv_score_pipeline(pipeline, X_train, y_train, cv=cv_split)
    except Exception:
        return 0.0
    return score

# -------------------------
# Run Optuna studies
# -------------------------
sampler = TPESampler(seed=RANDOM_STATE)

print("\nStarting Optuna study for XGBoost...")
study_xgb = optuna.create_study(direction="maximize", sampler=sampler, study_name="xgb_f1_macro")
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, show_progress_bar=True)

print("\nStarting Optuna study for GradientBoosting...")
study_gb = optuna.create_study(direction="maximize", sampler=sampler, study_name="gb_f1_macro")
study_gb.optimize(objective_gb, n_trials=N_TRIALS, show_progress_bar=True)

# Save studies
joblib.dump(study_xgb, os.path.join(MODELS_DIR, "study_xgb.pkl"))
joblib.dump(study_gb, os.path.join(MODELS_DIR, "study_gb.pkl"))

# -------------------------
# Fit best models on full training set and evaluate on test set
# -------------------------
results = []

def fit_and_eval_best_xgb():
    best_params = study_xgb.best_trial.params.copy()
    # add required fixed params
    best_params.update({"use_label_encoder": False, "objective": "multi:softprob", "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1})
    print("\nBest XGBoost params:", best_params)
    clf = XGBClassifier(**best_params)
    pipeline = build_pipeline(clf)
    t0 = time()
    pipeline.fit(X_train, y_train)
    tt = time() - t0
    joblib.dump(pipeline, os.path.join(MODELS_DIR, "XGBoost_best.pkl"))
    # eval
    y_pred = pipeline.predict(X_test)
    try:
        y_proba = pipeline.predict_proba(X_test)
    except Exception:
        y_proba = None
    report_dict = classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred), output_dict=True, zero_division=0)
    conf = confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(y_pred), labels=list(le.classes_))
    results.append({
        "model": "XGBoost",
        "best_params": best_params,
        "train_time_s": round(tt, 3),
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "test_precision_macro": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "test_recall_macro": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "test_roc_auc_macro": round(float(np.nan) if y_proba is None else float(__import__("sklearn").metrics.roc_auc_score(pd.get_dummies(y_test), y_proba, average="macro", multi_class="ovr")), 4) if y_proba is not None else None,
        "report": report_dict,
        "confusion_matrix": conf
    })

def fit_and_eval_best_gb():
    best_params = study_gb.best_trial.params.copy()
    print("\nBest GradientBoosting params:", best_params)
    clf = GradientBoostingClassifier(**best_params, random_state=RANDOM_STATE)
    pipeline = build_pipeline(clf)
    t0 = time()
    pipeline.fit(X_train, y_train)
    tt = time() - t0
    joblib.dump(pipeline, os.path.join(MODELS_DIR, "GradientBoosting_best.pkl"))
    # eval
    y_pred = pipeline.predict(X_test)
    try:
        y_proba = pipeline.predict_proba(X_test)
    except Exception:
        y_proba = None
    report_dict = classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred), output_dict=True, zero_division=0)
    conf = confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(y_pred), labels=list(le.classes_))
    results.append({
        "model": "GradientBoosting",
        "best_params": best_params,
        "train_time_s": round(tt, 3),
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "test_precision_macro": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "test_recall_macro": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "test_roc_auc_macro": round(float(np.nan) if y_proba is None else float(__import__("sklearn").metrics.roc_auc_score(pd.get_dummies(y_test), y_proba, average="macro", multi_class="ovr")), 4) if y_proba is not None else None,
        "report": report_dict,
        "confusion_matrix": conf
    })

fit_and_eval_best_xgb()
fit_and_eval_best_gb()

# Save summary CSV (without nested objects)
summary_rows = []
for r in results:
    summary_rows.append({
        "model": r["model"],
        "test_accuracy": r["test_accuracy"],
        "test_f1_macro": r["test_f1_macro"],
        "test_precision_macro": r["test_precision_macro"],
        "test_recall_macro": r["test_recall_macro"],
        "test_roc_auc_macro": r["test_roc_auc_macro"],
        "train_time_s": r["train_time_s"],
        "model_path": os.path.join(MODELS_DIR, f"{r['model']}_best.pkl")
    })
pd.DataFrame(summary_rows).to_csv(os.path.join(MODELS_DIR, "tuning_results_optuna.csv"), index=False)

# Print human readable reports
for r in results:
    print(f"\n\n=== {r['model']} results ===")
    print("Test accuracy:", r["test_accuracy"], "Test F1 macro:", r["test_f1_macro"], "ROC-AUC:", r["test_roc_auc_macro"])
    print("Classification report:")
    print(pd.DataFrame(r["report"]).transpose().to_string())
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(r["confusion_matrix"])

print("\nSaved best pipelines and studies into", MODELS_DIR)
