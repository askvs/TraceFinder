
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Get the absolute path to the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to the project root from the script's directory (codes/baseline -> codes -> myproject)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Construct the full path to the CSV file
CSV_PATH = os.path.join(project_root, "processed_data", "combined_features.csv")

def train_models():
    df = pd.read_csv(CSV_PATH)
    print("DataFrame Columns:", df.columns)
    X = df.drop(columns=["class_label"])
    y = df["class_label"]

    # Identify non-numeric columns and drop them
    non_numeric_cols = X.select_dtypes(include=['object']).columns
    if not non_numeric_cols.empty:
        print(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
        X = X.drop(columns=non_numeric_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(project_root, "models", "random_forest.pkl"))

    # Train SVM
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    svm.fit(X_train, y_train)
    joblib.dump(svm, os.path.join(project_root, "models", "svm.pkl"))

    
    joblib.dump(scaler, os.path.join(project_root, "models", "scaler.pkl"))

    print(" Models trained and saved successfully!")

if __name__ == "__main__":
    train_models()