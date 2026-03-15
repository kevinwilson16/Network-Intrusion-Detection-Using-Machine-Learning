import os
import joblib
import pandas as pd
import time
import json
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
PROCESSED_DATA_PATH = "data/cicids2017/processed"
MODELS_PATH = "artifacts/models"
METRICS_PATH = "artifacts/metrics"

def load_data():
    print("Loading multiclass data...")
    train_df = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "train_multiclass.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "test_multiclass.parquet"))
    
    X_train = train_df.drop(columns=['multiclass_label', 'label'])
    y_train = train_df['multiclass_label']
    
    X_test = test_df.drop(columns=['multiclass_label', 'label'])
    y_test = test_df['multiclass_label']
    
    return X_train, y_train, X_test, y_test

def train_and_eval(name, model, X_train, y_train, X_test, y_test, target_names, sample_weight=None):
    print(f"--- Experiment: {name} ---")
    start_time = time.time()
    
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
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
    
    X_train_raw, y_train_raw, X_test, y_test = load_data()
    le = joblib.load(os.path.join(MODELS_PATH, "multiclass_label_encoder.pkl"))
    target_names = le.classes_
    
    results = []
    limitations = []

    # --- 1. Standardize the Baseline ---
    # To ensure academic validity, both models must start from the exact same training footprint.
    print("Standardizing baseline: Downsampling majority class (BENIGN) to 100k...")
    counts = y_train_raw.value_counts()
    majority_class_idx = counts.idxmax()
    
    mask_majority = (y_train_raw == majority_class_idx)
    X_train_maj = X_train_raw[mask_majority]
    y_train_maj = y_train_raw[mask_majority]
    
    X_train_min = X_train_raw[~mask_majority]
    y_train_min = y_train_raw[~mask_majority]
    
    # Standardized Downsample
    X_maj_down = X_train_maj.sample(n=100000, random_state=42)
    y_maj_down = y_train_maj.loc[X_maj_down.index]
    
    X_train_base = pd.concat([X_maj_down, X_train_min])
    y_train_base = pd.concat([y_maj_down, y_train_min])
    
    print(f"Standardized Baseline Size: {len(X_train_base)} (Samples per class logged in metrics)")

    # --- 2. XGBoost with Balanced Sample Weights ---
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', y_train_base)
    xgb_balanced = XGBClassifier(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42, tree_method='hist')
    res_xgb_bal = train_and_eval("XGB_BalancedWeights", xgb_balanced, X_train_base, y_train_base, X_test, y_test, target_names, sample_weight=sample_weights)
    results.append(res_xgb_bal)
    joblib.dump(xgb_balanced, os.path.join(MODELS_PATH, "xgb_multiclass_balanced.pkl"))

    # --- 3. XGBoost with SMOTE ---
    # Filter classes with < 6 samples to allow k_neighbors=5
    print("Preparing data for SMOTE: Filtering ultra-rare classes...")
    base_counts = y_train_base.value_counts()
    keep_classes = base_counts[base_counts >= 6].index
    dropped_classes = base_counts[base_counts < 6].index
    
    if len(dropped_classes) > 0:
        dropped_names = [target_names[i] for i in dropped_classes]
        limitations.append(f"Excluded from SMOTE due to <6 samples: {dropped_names}")
        print(f"Dropped for SMOTE: {dropped_names}")

    X_train_smote_input = X_train_base[y_train_base.isin(keep_classes)]
    y_train_smote_input = y_train_base[y_train_base.isin(keep_classes)]

    print(f"Applying SMOTE on filtered dataset (Size: {len(X_train_smote_input)})...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    
    try:
        X_res, y_res = smote.fit_resample(X_train_smote_input, y_train_smote_input)
        print(f"SMOTE Counts: {len(X_train_smote_input)} -> {len(X_res)}")
        
        xgb_smote = XGBClassifier(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42, tree_method='hist')
        res_xgb_smote = train_and_eval("XGB_SMOTE", xgb_smote, X_res, y_res, X_test, y_test, target_names)
        results.append(res_xgb_smote)
        joblib.dump(xgb_smote, os.path.join(MODELS_PATH, "xgb_multiclass_smote.pkl"))
    except Exception as e:
        print(f"SMOTE experiment failed: {e}")

    # Output results and metadata
    output = {
        "results": results,
        "experiment_metadata": {
            "standardized_base_size": len(X_train_base),
            "limitations": limitations,
            "training_class_distribution": base_counts.to_dict()
        }
    }
    
    with open(os.path.join(METRICS_PATH, "multiclass_results.json"), "w") as f:
        json.dump(output, f, indent=4)
    
    print("Phase 3 Refactored Training Complete.")

if __name__ == "__main__":
    main()

