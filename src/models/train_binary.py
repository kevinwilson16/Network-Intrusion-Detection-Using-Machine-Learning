import os
import joblib
import pandas as pd
import numpy as np
import time
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, average_precision_score

# Paths
PROCESSED_DATA_PATH = "data/cicids2017/processed"
MODELS_PATH = "artifacts/models"
METRICS_PATH = "artifacts/metrics"

def load_data():
    print("Loading CIC-IDS2017 preprocessed binary data...")
    train_df = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "test.parquet"))
    
    # Features
    X_train = train_df.drop(columns=['is_attack', 'label'])
    y_train = train_df['is_attack']
    
    X_test = test_df.drop(columns=['is_attack', 'label'])
    y_test = test_df['is_attack']
    
    return X_train, y_train, X_test, y_test

def train_and_eval_binary(name, model, X_train, y_train, X_test, y_test):
    print(f"--- Training: {name} ---")
    start_time = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start_time
    
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    # Calculate AUC-PR
    auc_pr = average_precision_score(y_test, y_probs)
    
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC-PR: {auc_pr:.4f} | Time: {duration:.2f}s")
    
    return {
        "model": name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_pr": auc_pr,
        "duration": duration,
        "report": classification_report(y_test, y_pred, output_dict=True)
    }

def main():
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(METRICS_PATH, exist_ok=True)
    
    X_train, y_train, X_test, y_test = load_data()
    
    results = []

    # 1. Logistic Regression (Balanced)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1)
    res_lr = train_and_eval_binary("Logistic Regression", lr, X_train, y_train, X_test, y_test)
    results.append(res_lr)
    joblib.dump(lr, os.path.join(MODELS_PATH, "lr_binary.pkl"))

    # 2. Random Forest (Balanced, Limited Depth)
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    res_rf = train_and_eval_binary("Random Forest", rf, X_train, y_train, X_test, y_test)
    results.append(res_rf)
    joblib.dump(rf, os.path.join(MODELS_PATH, "rf_binary.pkl"))

    # Save Metrics
    with open(os.path.join(METRICS_PATH, "binary_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nPhase CIC-IDS2017 Binary Baseline Complete. Results saved to {METRICS_PATH}")

if __name__ == "__main__":
    main()
