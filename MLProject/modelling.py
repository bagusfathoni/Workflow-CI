import argparse
import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Parsing Argument (Agar bisa diatur via MLproject)
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
args = parser.parse_args()

# 2. Load Data
# Menggunakan relative path agar aman di berbagai env
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "diabetes_preprocessing.csv")

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: File {data_path} tidak ditemukan.")
    exit(1)

# 3. Data Splitting
# Sesuaikan dengan dataset Diabetes: Target = 'Outcome'
target_col = 'Outcome'

if target_col not in df.columns:
    print(f"Error: Kolom target '{target_col}' tidak ditemukan.")
    exit(1)

X = df.drop([target_col], axis=1) # Tidak perlu drop age_group jika tidak ada
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Training dengan MLflow
# Note: Jangan set tracking URI ke HTTP localhost untuk CI di GitHub
# Biarkan default (file:./mlruns) agar artifact tersimpan di workspace CI

mlflow.set_experiment("CI_Diabetes_Training")

with mlflow.start_run():
    # Init Model dengan parameter dari args
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Logging Parameters
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    
    # Logging Metrics
    mlflow.log_metric("accuracy", acc)

    # Logging Model (PENTING: Gunakan nama 'model' agar standar)
    mlflow.sklearn.log_model(model, "model")

    print(f"Training Selesai. Accuracy: {acc:.4f}")
    print(f"Model saved to mlflow artifacts.")