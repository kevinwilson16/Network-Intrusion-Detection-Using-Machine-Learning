import os
import joblib
import pandas as pd
import numpy as np
import time
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score, precision_recall_curve, auc

# Paths
PROCESSED_DATA_PATH = "data/unsw-nb15/processed"
MODELS_PATH = "artifacts/models"
METRICS_PATH = "artifacts/metrics"

def load_data():
    print("Loading UNSW-NB15 preprocessed binary data...")
    train_df = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "train_unsw.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "test_unsw.parquet"))
    
    # Features (Same as multiclass - ensure consistency)
    X_train = train_df.drop(columns=['multiclass_label', 'label'])
    y_train = train_df['label']
    
    X_test = test_df.drop(columns=['multiclass_label', 'label'])
    y_test = test_df['label']
    
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
    
    # Calculate AUC-PR (Area Under Precision-Recall Curve)
    p, r, _ = precision_recall_curve(y_test, y_probs)
    auc_pr = auc(r, p)
    
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC-PR: {auc_pr:.4f} | Time: {duration:.2f}s")
    
    return {
        "name": name,
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
    res_lr = train_and_eval_binary("LogisticRegression_UNSW", lr, X_train, y_train, X_test, y_test)
    results.append(res_lr)
    joblib.dump(lr, os.path.join(MODELS_PATH, "lr_unsw_binary.pkl"))

    # 2. Random Forest (Balanced, Limited Depth)
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    res_rf = train_and_eval_binary("RandomForest_UNSW", rf, X_train, y_train, X_test, y_test)
    results.append(res_rf)
    joblib.dump(rf, os.path.join(MODELS_PATH, "rf_unsw_binary.pkl"))

    # Save Metrics
    with open(os.path.join(METRICS_PATH, "unsw_binary_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nPhase 6 UNSW-NB15 Binary Baseline Complete. Results saved to {METRICS_PATH}")

if __name__ == "__main__":
    main()
