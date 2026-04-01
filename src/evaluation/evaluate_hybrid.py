import os
import joblib
import pandas as pd
import numpy as np
import time
import json
from sklearn.metrics import classification_report, accuracy_score

# Paths
PROCESSED_DATA_PATH = "data/cicids2017/processed"
MODELS_PATH = "artifacts/models"
METRICS_PATH = "artifacts/metrics"

def load_models_and_data():
    print("Loading pre-trained models and test dataset...")
    # Load Models (No retraining involved - prevents leakage)
    iso_forest = joblib.load(os.path.join(MODELS_PATH, "isolation_forest_base.pkl"))
    xgb_smote = joblib.load(os.path.join(MODELS_PATH, "xgb_multiclass_smote.pkl"))
    le = joblib.load(os.path.join(MODELS_PATH, "multiclass_label_encoder.pkl"))
    
    # Load Test Data only
    test_df = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "test_multiclass.parquet"))
    
    X_test = test_df.drop(columns=['multiclass_label', 'label'])
    y_test = test_df['multiclass_label']
    
    return iso_forest, xgb_smote, le, X_test, y_test

def main():
    os.makedirs(METRICS_PATH, exist_ok=True)
    
    iso_forest, xgb_smote, le, X_test, y_test = load_models_and_data()
    target_names = le.classes_
    
    print("Filtering test data to match XGBoost classes...")
    valid_classes = xgb_smote.classes_
    mask_valid = y_test.isin(valid_classes)
    X_test = X_test[mask_valid].reset_index(drop=True)
    y_test = y_test[mask_valid].reset_index(drop=True)
    
    # Identify the integer index for the 'BENIGN' class
    # Usually 0, but we locate it dynamically from the LabelEncoder
    benign_idx = list(le.classes_).index('BENIGN')
    print(f"BENIGN Label Index: {benign_idx}")

    start_time = time.time()
    
    print("\n--- STAGE 1: Isolation Forest (Triage) ---")
    # 1. Stage 1 Inference
    stage1_preds = iso_forest.predict(X_test)
    
    # 2. Generate Mask (-1 means Anomaly)
    anomaly_mask = (stage1_preds == -1)
    num_anomalies = np.sum(anomaly_mask)
    num_normal = len(X_test) - num_anomalies
    
    print(f"Total Test Samples: {len(X_test)}")
    print(f"Flagged as Anomalous (Routed to Stage 2): {num_anomalies} ({(num_anomalies/len(X_test))*100:.2f}%)")
    print(f"Passed as Normal (Bypassed Stage 2): {num_normal} ({(num_normal/len(X_test))*100:.2f}%)")

    # 3. Result Initialization
    # We initialize the final prediction array entirely with the 'BENIGN' class index
    final_preds = np.full(shape=len(X_test), fill_value=benign_idx, dtype=int)
    
    print("\n--- STAGE 2: XGBoost Classifier (Specific Attack Detection) ---")
    if num_anomalies > 0:
        # 4. Filter X_test using the mask
        X_test_flagged = X_test[anomaly_mask]
        
        # 5. Stage 2 Inference (Only on flagged data)
        stage2_preds = xgb_smote.predict(X_test_flagged)
        
        # 6. Reassembly (Inject predictions into the initialized array using the mask)
        final_preds[anomaly_mask] = stage2_preds
    else:
        print("No anomalies detected by Stage 1. Stage 2 bypassed completely.")

    duration = time.time() - start_time
    
    # Final Evaluation
    print("\n--- HYBRID PIPELINE EVALUATION ---")
    acc = accuracy_score(y_test, final_preds)
    filtered_target_names = [target_names[i] for i in valid_classes]
    report = classification_report(y_test, final_preds, labels=valid_classes, target_names=filtered_target_names, output_dict=True, zero_division=0)
    
    print(f"Hybrid Accuracy: {acc:.4f} | Total Inference Time: {duration:.2f}s")
    
    # Save Results
    output = {
        "experiment_name": "Two-Stage Hybrid Pipeline (IF -> XGBoost)",
        "accuracy": acc,
        "inference_duration_seconds": duration,
        "stage1_anomalies_flagged": int(num_anomalies),
        "stage1_bypassed": int(num_normal),
        "report": report
    }
    
    with open(os.path.join(METRICS_PATH, "hybrid_results.json"), "w") as f:
        json.dump(output, f, indent=4)
        
    print(f"Phase 5 Hybrid Evaluation Complete. Results saved to {METRICS_PATH}")

if __name__ == "__main__":
    main()
