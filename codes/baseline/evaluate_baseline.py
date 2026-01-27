import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Get the absolute path to the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Construct the full path to the CSV file
CSV_PATH = os.path.join(project_root, "processed_data", "combined_features.csv")

def evaluate_model(model_name, name, save_dir=None):
    # Load dataset
    df = pd.read_csv(CSV_PATH)
    X = df.drop(columns=["class_label"])
    y = df["class_label"]
    
    # Identify non-numeric columns and drop them
    non_numeric_cols = X.select_dtypes(include=['object']).columns
    if not non_numeric_cols.empty:
        print(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
        X = X.drop(columns=non_numeric_cols)

    # Load scaler + model
    scaler_path = os.path.join(project_root, "models", "scaler.pkl")
    scaler = joblib.load(scaler_path)
    
    model_path = os.path.join(project_root, "models", model_name)
    model = joblib.load(model_path)

    # Transform features
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    # Print report
    print(f"\n=== {name} Evaluation ===")
    print(classification_report(y, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=model.classes_,
                yticklabels=model.classes_,
                cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Set default save directory for baseline models
    if save_dir is None:
        save_dir = os.path.join(project_root, "results", "baseline")
    
    # Ensure results directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name.replace(' ', '_')}_confusion_matrix.png")

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f" Confusion matrix saved to: {save_path}")

    # Show plot (optional, you can comment this out if not needed)
    plt.show()

if __name__ == "__main__":
    evaluate_model("random_forest.pkl", "Random Forest")
    evaluate_model("svm.pkl", "SVM")
