import os
import joblib
import pandas as pd
import numpy as np
import time
import json
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

# Paths
PROCESSED_DATA_PATH = "data/unsw-nb15/processed"
MODELS_PATH = "artifacts/models"
METRICS_PATH = "artifacts/metrics"

def load_data():
    print("Loading UNSW-NB15 preprocessed data...")
    train_df = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "train_unsw.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "test_unsw.parquet"))
    
    X_train = train_df.drop(columns=['multiclass_label', 'label'])
    y_train = train_df['multiclass_label']
    y_train_binary = train_df['label']
    
    X_test = test_df.drop(columns=['multiclass_label', 'label'])
    y_test = test_df['multiclass_label']
    y_test_binary = test_df['label']
    
    return X_train, y_train, y_train_binary, X_test, y_test, y_test_binary

def train_and_eval(name, model, X_train, y_train, X_test, y_test, target_names):
    print(f"--- Experiment: {name} ---")
    start_time = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start_time
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    
    print(f"Accuracy: {acc:.4f} | Time: {duration:.2f}s")
    return {"name": name, "accuracy": acc, "report": report, "duration": duration}

def main():
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(METRICS_PATH, exist_ok=True)
    
    X_train_raw, y_train_raw, y_train_bin_raw, X_test, y_test, y_test_bin = load_data()
    
    # Label Encoding for Multiclass
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_raw)
    y_test_enc = le.transform(y_test)
    target_names = le.classes_
    joblib.dump(le, os.path.join(MODELS_PATH, "unsw_label_encoder.pkl"))
    
    results = []
    limitations = []

    # --- 1. Standardize the Baseline ---
    # Downsample 'Normal' to 100k for academic consistency
    print("Standardizing baseline: Downsampling 'Normal' to 100k...")
    normal_label = 'Normal'
    mask_normal = (y_train_raw == normal_label)
    
    X_train_norm = X_train_raw[mask_normal]
    
    # Set downsample size
    downsample_size = min(100000, len(X_train_norm))
    X_norm_down = X_train_norm.sample(n=downsample_size, random_state=42)
    
    # Extract labels for both groups
    # We use the fact that y_train_enc aligns with X_train_raw.
    # To get y_norm_down_enc, we mask y_train_enc with the downsampled indices.
    mask_norm_down = X_train_raw.index.isin(X_norm_down.index)
    mask_attack = ~mask_normal
    
    X_train_base = pd.concat([X_norm_down, X_train_raw[mask_attack]])
    y_train_base = np.concatenate([y_train_enc[mask_norm_down], y_train_enc[mask_attack]])
    
    print(f"Standardized Baseline Size: {len(X_train_base)}")


    # --- 2. Random Forest (Balanced Weights) ---
    rf_balanced = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42, class_weight='balanced')
    res_rf = train_and_eval("RF_BalancedWeights_UNSW", rf_balanced, X_train_base, y_train_base, X_test, y_test_enc, target_names)
    results.append(res_rf)
    joblib.dump(rf_balanced, os.path.join(MODELS_PATH, "rf_unsw_balanced.pkl"))

    # --- 3. XGBoost (SMOTE) ---
    print("Preparing SMOTE: Filtering ultra-rare classes (<6 samples)...")
    unique, counts = np.unique(y_train_base, return_counts=True)
    keep_classes = unique[counts >= 6]
    dropped_classes = unique[counts < 6]
    
    if len(dropped_classes) > 0:
        dropped_names = [target_names[i] for i in dropped_classes]
        limitations.append(f"Excluded from SMOTE: {dropped_names}")
        print(f"Dropped for SMOTE: {dropped_names}")

    mask_smote = np.isin(y_train_base, keep_classes)
    X_smote_in = X_train_base[mask_smote]
    y_smote_in = y_train_base[mask_smote]

    smote = SMOTE(random_state=42, k_neighbors=5)
    try:
        X_res, y_res = smote.fit_resample(X_smote_in, y_smote_in)
        xgb_smote = XGBClassifier(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42, tree_method='hist')
        res_xgb = train_and_eval("XGB_SMOTE_UNSW", xgb_smote, X_res, y_res, X_test, y_test_enc, target_names)
        results.append(res_xgb)
        joblib.dump(xgb_smote, os.path.join(MODELS_PATH, "xgb_unsw_smote.pkl"))
    except Exception as e:
        print(f"SMOTE failed: {e}")

    # --- 4. True Unsupervised (Isolation Forest) ---
    print("\n--- Experiment: Isolation Forest (True Unsupervised) ---")
    iso_forest = IsolationForest(n_estimators=100, contamination=0.15, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train_base) # Mixed dataset, no labels
    
    preds_raw = iso_forest.predict(X_test)
    # Map -1 (Anomaly) to 1 (Attack), 1 (Normal) to 0 (Normal) for binary eval
    y_pred_unsub = np.where(preds_raw == -1, 1, 0)
    y_test_bin = np.where(y_test != 'Normal', 1, 0)

    
    prec, rec, f1, _ = precision_recall_fscore_support(y_test_bin, y_pred_unsub, average='binary', zero_division=0)
    print(f"Unsupervised (Binary) Metrics: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    
    joblib.dump(iso_forest, os.path.join(MODELS_PATH, "iso_forest_unsw.pkl"))

    # --- 5. Hybrid Evaluation ---
    print("\n--- Experiment: Hybrid (IF -> XGB) ---")
    # Identify indices flagged as anomalies by Stage 1
    anomaly_mask = (preds_raw == -1)
    
    # Initialize final predictions with 'Normal' index
    normal_idx = list(target_names).index('Normal')
    final_hybrid_preds = np.full(len(X_test), fill_value=normal_idx)
    
    if np.any(anomaly_mask):
        X_test_flagged = X_test[anomaly_mask]
        # Use our saved XGB model
        try:
           xgb_model = joblib.load(os.path.join(MODELS_PATH, "xgb_unsw_smote.pkl"))
           stage2_preds = xgb_model.predict(X_test_flagged)
           final_hybrid_preds[anomaly_mask] = stage2_preds
        except:
           print("Hybrid Stage 2 skipped (XGB model missing)")

    acc_hyb = accuracy_score(y_test_enc, final_hybrid_preds)
    report_hyb = classification_report(y_test_enc, final_hybrid_preds, target_names=target_names, output_dict=True, zero_division=0)
    print(f"Hybrid Accuracy: {acc_hyb:.4f}")

    # Output JSON Metrics
    output = {
        "supervised": results,
        "unsupervised": {"precision": prec, "recall": rec, "f1": f1},
        "hybrid": {"accuracy": acc_hyb, "report": report_hyb},
        "metadata": {"limitations": limitations}
    }
    
    with open(os.path.join(METRICS_PATH, "unsw_results.json"), "w") as f:
        json.dump(output, f, indent=4)
    
    print("Phase 6 UNSW-NB15 Pipeline Complete.")

if __name__ == "__main__":
    main()
