#!/usr/bin/env python3
"""
train_with_adasyn_xgboost_labelencode.py

Trains multiple classifiers using ADASYN oversampling and includes XGBoost.
Encodes string labels to integers via LabelEncoder to avoid XGBoost errors.
Saves trained pipelines, the label encoder and a CSV summary into models/.
"""

import os
import joblib
import time
from time import time as now
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# xgboost
from xgboost import XGBClassifier

# imbalanced-learn
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

CSV_PATH = "processed_data/merged_metadata.csv"
MODELS_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

def make_base_classifiers():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs"),
        "KNeighbors": KNeighborsClassifier(n_neighbors=5),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVC": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        classes = np.unique(y_true)
        if len(classes) == 2:
            if y_score.ndim == 2 and y_score.shape[1] == 2:
                return float(roc_auc_score(y_true, y_score[:, 1]))
            else:
                return float(roc_auc_score(y_true, y_score))
        else:
            lb = LabelBinarizer()
            Y = lb.fit_transform(y_true)
            return float(roc_auc_score(Y, y_score, average="macro", multi_class="ovr"))
    except Exception:
        return float("nan")

def build_pipeline(clf: Any):
    steps = [
        ("scaler", StandardScaler()),
        ("adasyn", ADASYN(random_state=RANDOM_STATE)),
        ("clf", clf),
    ]
    return ImbPipeline(steps)

def train_evaluate_save():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    if "class_label" not in df.columns:
        raise KeyError("CSV must contain 'class_label' column.")

    # Keep original labels for reporting, then encode
    y_raw = df["class_label"].astype(str).copy()
    X = df.drop(columns=["file_name", "main_class", "resolution", "class_label"], errors="ignore")

    # Label encode y -> integers 0..n_classes-1
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)  # numpy array
    # Save encoder
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

    print("Original classes:", list(le.classes_))
    print("Class counts (original):")
    print(pd.Series(y_raw).value_counts().sort_index())
    print()

    # Stratified split using encoded labels
    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, stratify=y_encoded, random_state=RANDOM_STATE
    )

    # For printing readable train/test distributions convert back to label names
    y_train_names = le.inverse_transform(y_train_enc)
    y_test_names = le.inverse_transform(y_test_enc)
    print(f"Train samples: {len(y_train_enc)}, Test samples: {len(y_test_enc)}")
    print("Train class distribution:")
    print(pd.Series(y_train_names).value_counts().sort_index())
    print()

    classifiers = make_base_classifiers()
    results = []
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, base_clf in classifiers.items():
        print(f"\n--- {name} ---")
        pipeline = build_pipeline(base_clf)

        # cross_val_score expects y encoded as integers (we pass y_train_enc)
        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train_enc, cv=skf, scoring="f1_macro", n_jobs=-1)
            cv_f1 = float(np.mean(cv_scores))
        except Exception as e:
            print(f"  CV failed for {name}: {e}")
            cv_f1 = float("nan")

        # Fit pipeline on training set
        t0 = now()
        pipeline.fit(X_train, y_train_enc)
        train_time = now() - t0

        # Save pipeline
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(pipeline, model_path)

        # Evaluate on test set (encoded labels)
        y_pred_enc = pipeline.predict(X_test)
        acc = accuracy_score(y_test_enc, y_pred_enc)
        prec = precision_score(y_test_enc, y_pred_enc, average="macro", zero_division=0)
        rec = recall_score(y_test_enc, y_pred_enc, average="macro", zero_division=0)
        f1 = f1_score(y_test_enc, y_pred_enc, average="macro", zero_division=0)

        # Probabilities for ROC-AUC (work with encoded labels)
        roc_auc = float("nan")
        try:
            if hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(X_test)
                roc_auc = safe_roc_auc(y_test_enc, np.asarray(y_proba))
            else:
                if hasattr(pipeline, "decision_function"):
                    y_score = pipeline.decision_function(X_test)
                    roc_auc = safe_roc_auc(y_test_enc, np.asarray(y_score))
        except Exception:
            roc_auc = float("nan")

        # Convert encoded predictions back to original labels for a readable report
        y_pred_names = le.inverse_transform(y_pred_enc)
        y_test_names = le.inverse_transform(y_test_enc)

        report = classification_report(y_test_names, y_pred_names, output_dict=True, zero_division=0)
        confmat = confusion_matrix(y_test_names, y_pred_names, labels=list(le.classes_))

        results.append({
            "model": name,
            "cv_f1_macro": round(cv_f1, 4) if not np.isnan(cv_f1) else np.nan,
            "test_accuracy": round(acc, 4),
            "test_precision_macro": round(prec, 4),
            "test_recall_macro": round(rec, 4),
            "test_f1_macro": round(f1, 4),
            "test_roc_auc_macro": round(roc_auc, 4) if not np.isnan(roc_auc) else np.nan,
            "train_time_s": round(train_time, 3),
            "model_path": model_path,
            "report": report,
            "confusion_matrix": confmat,
        })

        print(f"Trained {name} in {train_time:.2f}s | test f1_macro={f1:.4f} | roc_auc={roc_auc if not np.isnan(roc_auc) else 'n/a'}")

    # Summary
    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values(by="test_f1_macro", ascending=False).reset_index(drop=True)
    print("\n\n=== Summary (sorted by test F1 macro) ===")
    display_cols = ["model", "cv_f1_macro", "test_accuracy", "test_f1_macro", "test_roc_auc_macro", "train_time_s", "model_path"]
    print(results_df_sorted[display_cols].to_string(index=False))

    # Top-3 detailed reports
    top_k = min(3, len(results_df_sorted))
    for i in range(top_k):
        row = results_df_sorted.iloc[i]
        print(f"\n\n*** Top {i+1}: {row['model']} ***")
        print("Classification report:")
        print(pd.DataFrame(row["report"]).transpose().to_string())
        print("\nConfusion matrix (rows=true, cols=predicted):")
        print(row["confusion_matrix"])

    # Save CSV summary
    results_df_sorted.to_csv(os.path.join(MODELS_DIR, "model_comparison_results_adasyn_xgboost_labelencoded.csv"), index=False)
    print(f"\nSaved pipelines, label encoder and summary CSV in '{MODELS_DIR}/'.")

if __name__ == "__main__":
    train_evaluate_save()
