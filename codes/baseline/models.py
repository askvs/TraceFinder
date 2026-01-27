#!/usr/bin/env python3
"""
train_compare_models.py

Train and compare 10 classifiers (including RandomForest and SVC) on your dataset,
save trained models and scaler to models/ and print comparison results.

Usage:
    python train_compare_models.py
"""

import os
import joblib
import pandas as pd
import numpy as np
from time import time
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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
from sklearn.naive_bayes import GaussianNB

CSV_PATH = "processed_data/merged_metadata.csv"
MODELS_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

def make_models():
    """
    Return a dict of name -> untrained sklearn estimator.
    Contains 10 classifiers (including RandomForest and SVC).
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs", multi_class="auto"),
        "KNeighbors": KNeighborsClassifier(n_neighbors=5),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVC": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_STATE),
        # Bonus: if you want Naive Bayes uncomment next line
        # "GaussianNB": GaussianNB(),
    }
    return models

def safe_roc_auc(y_true, y_score, average="macro"):
    """
    Compute ROC AUC in a safe way that handles binary/multiclass and estimators
    that do not provide probabilities.
    y_score should be either probability estimates or decision function.
    """
    try:
        if len(np.unique(y_true)) == 2:
            # binary
            return roc_auc_score(y_true, y_score[:, 1] if y_score.ndim > 1 else y_score)
        else:
            # multiclass - one-vs-rest with macro averaging
            lb = LabelBinarizer()
            Y = lb.fit_transform(y_true)
            # If label binarizer returns shape (n_samples,1) for binary still handle
            if Y.shape[1] == 1:
                return roc_auc_score(y_true, y_score[:, 1] if y_score.ndim > 1 else y_score)
            return roc_auc_score(Y, y_score, average=average, multi_class="ovr")
    except Exception:
        return np.nan

def train_and_evaluate():
    # Load data
    df = pd.read_csv(CSV_PATH)
    if "class_label" not in df.columns:
        raise KeyError("'class_label' column not found in CSV.")
    X = df.drop(columns=["file_name", "main_class", "resolution", "class_label"], errors="ignore")
    y = df["class_label"].copy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Preprocess
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ensure models dir
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    models = make_models()

    results = []
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        t0 = time()
        # Fit on training set
        clf.fit(X_train_scaled, y_train)
        train_time = time() - t0

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(clf, model_path)

        # Cross-validated F1 (macro) on training data
        try:
            cv_f1 = cross_val_score(clf, X_train_scaled, y_train, cv=skf, scoring="f1_macro", n_jobs=-1).mean()
        except Exception:
            cv_f1 = np.nan

        # Predict on test set
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # ROC AUC if possible
        y_score = None
        roc_auc = np.nan
        if hasattr(clf, "predict_proba"):
            try:
                y_score = clf.predict_proba(X_test_scaled)
                roc_auc = safe_roc_auc(y_test, y_score, average="macro")
            except Exception:
                roc_auc = np.nan
        elif hasattr(clf, "decision_function"):
            try:
                y_score = clf.decision_function(X_test_scaled)
                # decision_function for multiclass shape handling
                if y_score.ndim == 1:
                    roc_auc = safe_roc_auc(y_test, y_score, average="macro")
                else:
                    roc_auc = safe_roc_auc(y_test, y_score, average="macro")
            except Exception:
                roc_auc = np.nan

        # Save detailed reports for top models later
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        confmat = confusion_matrix(y_test, y_pred)

        results.append({
            "model": name,
            "train_time_s": round(train_time, 3),
            "cv_f1_macro": round(cv_f1, 4) if not np.isnan(cv_f1) else np.nan,
            "test_accuracy": round(acc, 4),
            "test_precision_macro": round(prec, 4),
            "test_recall_macro": round(rec, 4),
            "test_f1_macro": round(f1, 4),
            "test_roc_auc_macro": (round(roc_auc, 4) if not np.isnan(roc_auc) else np.nan),
            "model_path": model_path,
            "report": report,
            "confusion_matrix": confmat,
        })

        print(f"{name} trained in {train_time:.2f}s — test f1_macro: {f1:.4f}, roc_auc: {roc_auc if not np.isnan(roc_auc) else 'n/a'}")

    # Summarize results
    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values(by="test_f1_macro", ascending=False).reset_index(drop=True)

    print("\n\n=== Summary (sorted by test F1 macro) ===")
    display_cols = ["model", "cv_f1_macro", "test_accuracy", "test_f1_macro", "test_roc_auc_macro", "train_time_s", "model_path"]
    print(results_df_sorted[display_cols].to_string(index=False))

    # Print classification report and confusion matrix for top 3 models
    top_k = min(3, len(results_df_sorted))
    for i in range(top_k):
        row = results_df_sorted.iloc[i]
        print(f"\n\n*** Top {i+1}: {row['model']} ***")
        print("Classification report:")
        print(pd.DataFrame(row["report"]).transpose().to_string())
        print("\nConfusion matrix:")
        print(row["confusion_matrix"])

    # Save results table to CSV
    results_df_sorted.to_csv(os.path.join(MODELS_DIR, "model_comparison_results.csv"), index=False)
    print(f"\nAll trained models and scaler saved in '{MODELS_DIR}/'. Results table saved as 'model_comparison_results.csv'.")

if __name__ == "__main__":
    train_and_evaluate()
