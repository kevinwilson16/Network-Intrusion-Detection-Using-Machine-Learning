import os
import joblib
import pandas as pd
import numpy as np
import time
import json
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, average_precision_score
from sklearn.preprocessing import LabelEncoder, label_binarize

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

def train_and_eval(name, model, X_train, y_train, X_test, y_test, target_names, sample_weight=None):
    print(f"--- Experiment: {name} ---")
    start_time = time.time()
    
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
        
    duration = time.time() - start_time
    
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Calculate AUC-PR (Macro)
    classes = np.arange(len(target_names))
    y_test_bin = label_binarize(y_test, classes=classes)
    auc_pr = average_precision_score(y_test_bin, y_probs, average='macro')
    
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    
    print(f"Accuracy: {acc:.4f} | AUC-PR: {auc_pr:.4f} | Time: {duration:.2f}s")
    return {"name": name, "accuracy": acc, "auc_pr": auc_pr, "report": report, "duration": duration}

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


    # --- 2. XGBoost (Balanced Weights) ---
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights_unsw = compute_sample_weight('balanced', y_train_base)
    xgb_balanced = XGBClassifier(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42, tree_method='hist')
    res_xgb_bal = train_and_eval("XGB_BalancedWeights_UNSW", xgb_balanced, X_train_base, y_train_base, X_test, y_test_enc, target_names, sample_weight=sample_weights_unsw)
    results.append(res_xgb_bal)
    joblib.dump(xgb_balanced, os.path.join(MODELS_PATH, "xgb_unsw_balanced.pkl"))

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
        print(f"SMOTE Counts: {len(X_smote_in)} -> {len(X_res)}")
        
        # --- FIX: TEMPORARY LABEL REMAPPING ---
        uniq_labels = np.unique(y_res)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(uniq_labels)}
        inverse_mapping = {new_label: old_label for new_label, old_label in enumerate(uniq_labels)}
        
        y_res_mapped = np.array([label_mapping[y] for y in y_res])
        
        xgb_smote = XGBClassifier(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42, tree_method='hist')
        
        start_time = time.time()
        xgb_smote.fit(X_res, y_res_mapped)
        duration = time.time() - start_time
        
        y_pred_mapped = xgb_smote.predict(X_test)
        y_probs_mapped = xgb_smote.predict_proba(X_test)
        
        y_pred = np.array([inverse_mapping.get(y, y) for y in y_pred_mapped])
        
        y_probs = np.zeros((len(X_test), len(target_names)))
        for mapped_idx, orig_idx in inverse_mapping.items():
            y_probs[:, orig_idx] = y_probs_mapped[:, mapped_idx]
            
        y_test_bin = label_binarize(y_test_enc, classes=np.arange(len(target_names)))
        auc_pr = average_precision_score(y_test_bin, y_probs, average='macro')
        
        acc = accuracy_score(y_test_enc, y_pred)
        report = classification_report(y_test_enc, y_pred, target_names=target_names, output_dict=True, zero_division=0)
        
        res_xgb = {"name": "XGB_SMOTE_UNSW", "accuracy": acc, "auc_pr": auc_pr, "report": report, "duration": duration}
        results.append(res_xgb)
        joblib.dump(xgb_smote, os.path.join(MODELS_PATH, "xgb_unsw_smote.pkl"))
    except Exception as e:
        print(f"SMOTE failed: {e}")

    # Output JSON Metrics
    output = {
        "supervised": results,
        "metadata": {"limitations": limitations}
    }
    
    with open(os.path.join(METRICS_PATH, "unsw_multiclass_results.json"), "w") as f:
        json.dump(output, f, indent=4)
    
    print("UNSW-NB15 Multiclass Pipeline Complete.")

if __name__ == "__main__":
    main()
