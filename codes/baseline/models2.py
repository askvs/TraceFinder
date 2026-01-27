#!/usr/bin/env python3
"""
train_with_smote.py

Train and compare 10 classifiers using SMOTE in the pipeline.
Saves trained pipelines and a summary CSV into models/.
"""

import os
import joblib
import numpy as np
import pandas as pd
from time import time

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer 
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

# imbalanced-learn components
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

CSV_PATH = "processed_data/merged_metadata.csv"
MODELS_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

def make_base_classifiers():
    """Return a dict of name -> sklearn estimator (unwrapped)."""
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
    }

def safe_roc_auc(y_true, y_score):
    """Compute ROC AUC for binary/multiclass in a safe way.
    y_score should be probability estimates with shape (n_samples, n_classes) or (n_samples,) for binary.
    """
    try:
        classes = np.unique(y_true)
        if len(classes) == 2:
            # binary, y_score can be shape (n_samples, 2) or (n_samples,)
            if y_score.ndim == 2 and y_score.shape[1] == 2:
                return roc_auc_score(y_true, y_score[:, 1])
            else:
                return roc_auc_score(y_true, y_score)
        else:
            # multiclass: one-vs-rest macro average
            lb = LabelBinarizer()
            Y = lb.fit_transform(y_true)
            # if Y has shape (n_samples, ) for binary handled above, else multiclass
            return roc_auc_score(Y, y_score, average="macro", multi_class="ovr")
    except Exception:
        return np.nan

def build_pipeline(clf):
    """
    Build an imblearn Pipeline:
    StandardScaler -> SMOTE -> classifier
    """
    steps = [
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", clf),
    ]
    return ImbPipeline(steps)

def train_evaluate_save():
    # Create models dir
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load data
    df = pd.read_csv(CSV_PATH)
    if "class_label" not in df.columns:
        raise KeyError("CSV must contain 'class_label' column.")
    X = df.drop(columns=["file_name", "main_class", "resolution", "class_label"], errors="ignore")
    y = df["class_label"].copy()

    # Show class distribution (print)
    print("Class distribution (full dataset):")
    print(y.value_counts(dropna=False).sort_index())
    print()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
    print("Train class distribution:")
    print(y_train.value_counts())
    print()

    classifiers = make_base_classifiers()
    results = []

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, base_clf in classifiers.items():
        print(f"\n--- {name} ---")
        pipeline = build_pipeline(base_clf)

        # Cross-validated F1-macro on training set (SMOTE will be applied inside the CV via the pipeline)
        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
            cv_f1 = float(np.mean(cv_scores))
        except Exception as e:
            print(f"CV failed for {name}: {e}")
            cv_f1 = np.nan

        # Fit pipeline on full training set
        t0 = time()
        pipeline.fit(X_train, y_train)
        train_time = time() - t0

        # Save pipeline
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(pipeline, model_path)

        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Try to get probability estimates for ROC-AUC
        roc_auc = np.nan
        try:
            if hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(X_test)
                roc_auc = safe_roc_auc(y_test, y_proba)
            else:
                # fallback: try decision_function
                if hasattr(pipeline, "decision_function"):
                    y_score = pipeline.decision_function(X_test)
                    # if decision_function gives shape (n_samples,) for binary that's ok
                    roc_auc = safe_roc_auc(y_test, y_score)
        except Exception:
            roc_auc = np.nan

        # Detailed report and confmat
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        confmat = confusion_matrix(y_test, y_pred)

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

        print(f"Trained {name} in {train_time:.2f}s | test f1_macro={f1:.4f} | test_roc_auc={roc_auc if not np.isnan(roc_auc) else 'n/a'}")

    # Summarize
    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values(by="test_f1_macro", ascending=False).reset_index(drop=True)

    print("\n\n=== Summary (sorted by test F1 macro) ===")
    display_cols = ["model", "cv_f1_macro", "test_accuracy", "test_f1_macro", "test_roc_auc_macro", "train_time_s", "model_path"]
    print(results_df_sorted[display_cols].to_string(index=False))

    # Show classification report & confusion matrix for top 3
    top_k = min(3, len(results_df_sorted))
    for i in range(top_k):
        row = results_df_sorted.iloc[i]
        print(f"\n\n*** Top {i+1}: {row['model']} ***")
        print("Classification report:")
        print(pd.DataFrame(row["report"]).transpose().to_string())
        print("\nConfusion matrix:")
        print(row["confusion_matrix"])

    # Save results
    results_df_sorted.to_csv(os.path.join(MODELS_DIR, "model_comparison_results_smote.csv"), index=False)
    print(f"\nSaved pipelines to '{MODELS_DIR}/' and summary CSV 'model_comparison_results_smote.csv'.")

if __name__ == "__main__":
    train_evaluate_save()
