import os
import joblib
import pandas as pd
import numpy as np
import time
import json
from sklearn.metrics import classification_report, accuracy_score

# Paths
PROCESSED_DATA_PATH = "data/unsw-nb15/processed"
MODELS_PATH = "artifacts/models"
METRICS_PATH = "artifacts/metrics"

def load_data():
    print("Loading UNSW-NB15 preprocessed data...")
    test_df = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "test_unsw.parquet"))
    
    X_test = test_df.drop(columns=['multiclass_label', 'label'])
    y_test = test_df['multiclass_label']
    
    return X_test, y_test

def main():
    X_test_full, y_test_full = load_data()
    
    print("\n--- Experiment: Hybrid (IF -> XGB) on UNSW-NB15 ---")
    
    # Load Label Encoder
    try:
        le = joblib.load(os.path.join(MODELS_PATH, "unsw_label_encoder.pkl"))
        y_test_enc_full = le.transform(y_test_full)
        target_names = le.classes_
        normal_idx = list(target_names).index('Normal')
    except Exception as e:
        print(f"Error loading label encoder: {e}")
        return

    # Load Isolation Forest
    try:
        iso_forest = joblib.load(os.path.join(MODELS_PATH, "iso_forest_unsw.pkl"))
    except Exception as e:
        print(f"Error loading Isolation Forest model: {e}")
        return
        
    # Load XGB model to find valid classes
    try:
        xgb_model = joblib.load(os.path.join(MODELS_PATH, "xgb_unsw_smote.pkl"))
        valid_classes = xgb_model.classes_
    except Exception as e:
        print(f"Error loading XGB model: {e}")
        return

    print("Filtering test data to match XGBoost classes...")
    mask_valid = np.isin(y_test_enc_full, valid_classes)
    X_test = X_test_full[mask_valid].reset_index(drop=True)
    y_test_enc = y_test_enc_full[mask_valid]
    filtered_target_names = [target_names[i] for i in valid_classes]

    start_time = time.time()
    preds_raw = iso_forest.predict(X_test)
    anomaly_mask = (preds_raw == -1)
    
    final_hybrid_preds = np.full(len(X_test), fill_value=normal_idx)
    
    if np.any(anomaly_mask):
        X_test_flagged = X_test[anomaly_mask]
        stage2_preds = xgb_model.predict(X_test_flagged)
        final_hybrid_preds[anomaly_mask] = stage2_preds

    duration = time.time() - start_time
    
    acc_hyb = accuracy_score(y_test_enc, final_hybrid_preds)
    report_hyb = classification_report(y_test_enc, final_hybrid_preds, labels=valid_classes, target_names=filtered_target_names, output_dict=True, zero_division=0)
    print(f"Hybrid Accuracy: {acc_hyb:.4f} | Inference Time: {duration:.2f}s")
    
    # Track flagged metrics
    anomalies_flagged = np.sum(anomaly_mask)
    bypassed = len(X_test) - anomalies_flagged
    print(f"Stage 1 Anomalies Flagged: {anomalies_flagged}")
    print(f"Stage 1 Bypassed (Normal): {bypassed}")

    output = {
        "experiment_name": "Two-Stage Hybrid Pipeline (IF -> XGBoost) UNSW-NB15",
        "accuracy": acc_hyb,
        "inference_duration_seconds": duration,
        "stage1_anomalies_flagged": int(anomalies_flagged),
        "stage1_bypassed": int(bypassed),
        "report": report_hyb
    }
    
    with open(os.path.join(METRICS_PATH, "unsw_hybrid_results.json"), "w") as f:
        json.dump(output, f, indent=4)
        
    print("UNSW-NB15 Hybrid Pipeline Complete.")

if __name__ == "__main__":
    main()
