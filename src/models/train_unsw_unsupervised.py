import os
import joblib
import pandas as pd
import numpy as np
import time
import json
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Paths
PROCESSED_DATA_PATH = "data/unsw-nb15/processed"
MODELS_PATH = "artifacts/models"
METRICS_PATH = "artifacts/metrics"


def load_data():
    print("Loading UNSW-NB15 preprocessed data...")
    train_df = pd.read_parquet(os.path.join(
        PROCESSED_DATA_PATH, "train_unsw.parquet"))
    test_df = pd.read_parquet(os.path.join(
        PROCESSED_DATA_PATH, "test_unsw.parquet"))

    X_train = train_df.drop(columns=['multiclass_label', 'label'])
    y_train = train_df['multiclass_label']
    y_train_binary = train_df['label']

    X_test = test_df.drop(columns=['multiclass_label', 'label'])
    y_test = test_df['multiclass_label']
    y_test_binary = test_df['label']

    return X_train, y_train, y_train_binary, X_test, y_test, y_test_binary


def main():
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(METRICS_PATH, exist_ok=True)

    X_train_raw, y_train_raw, y_train_bin_raw, X_test, y_test, y_test_bin = load_data()

    print("\n--- Experiment: Isolation Forest (True Unsupervised) ---")
    start_time = time.time()

    iso_forest = IsolationForest(
        n_estimators=100, contamination=0.15, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train_raw)  # Mixed dataset, no labels

    duration = time.time() - start_time
    print(f"Isolation Forest trained in {duration:.2f}s")

    inference_start = time.time()
    preds_raw = iso_forest.predict(X_test)
    inference_duration = time.time() - inference_start
    print(f"Isolation Forest inference completed in {inference_duration:.4f}s")

    # Map -1 (Anomaly) to 1 (Attack), 1 (Normal) to 0 (Normal) for binary eval
    y_pred_unsub = np.where(preds_raw == -1, 1, 0)
    y_test_bin = np.where(y_test != 'Normal', 1, 0)

    acc = accuracy_score(y_test_bin, y_pred_unsub)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test_bin, y_pred_unsub, average='binary', zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred_unsub).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(
        f"Unsupervised (Binary) Metrics: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, FAR={far:.4f}")

    joblib.dump(iso_forest, os.path.join(MODELS_PATH, "iso_forest_unsw.pkl"))

    # Output JSON Metrics
    output = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "far": float(far),
        "inference_time_seconds": float(inference_duration)
    }

    with open(os.path.join(METRICS_PATH, "unsw_unsupervised_results.json"), "w") as f:
        json.dump(output, f, indent=4)

    print("UNSW-NB15 Unsupervised Pipeline Complete.")


if __name__ == "__main__":
    main()
