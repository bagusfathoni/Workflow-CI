import argparse
import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Parsing Argument
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
args = parser.parse_args()

# 2. Load Data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "diabetes_preprocessing.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"File tidak ditemukan: {data_path}")

df = pd.read_csv(data_path)

# 3. Data Splitting
target_col = "Outcome"
if target_col not in df.columns:
    raise ValueError(f"Kolom target '{target_col}' tidak ditemukan")

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Training & Logging
# PENTING: Gunakan start_run() agar aman dijalankan manual maupun via CI
with mlflow.start_run():
    # Train Model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluasi
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Logging Parameters
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    
    # Logging Metrics
    mlflow.log_metric("accuracy", acc)

    # Logging Model (Wajib untuk Advance/Docker)
    # Nama artifact_path="model" harus konsisten dengan langkah build-docker di YAML
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"✅ Training selesai | Accuracy: {acc:.4f}")