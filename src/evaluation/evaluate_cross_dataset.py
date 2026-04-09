import os
import pandas as pd
import numpy as np
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, average_precision_score, confusion_matrix

# Paths
CROSS_DATA_PATH = "data/cross_dataset"
METRICS_PATH = "artifacts/metrics"

def load_split_scale(train_path, test_path):
    """Loads datasets, splits training data, and applies strict scaler fitting."""
    # Load raw, mapped, unit-converted shared features
    df_train_full = pd.read_parquet(train_path)
    df_test_full = pd.read_parquet(test_path)
    
    # Extract features/labels
    X_train_full = df_train_full.drop(columns=['is_attack'])
    y_train_full = df_train_full['is_attack']
    
    X_test_ext = df_test_full.drop(columns=['is_attack'])
    y_test_ext = df_test_full['is_attack']
    
    # Internal validation split for the training dataset to prevent overfitting
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    X_train, X_val, X_test_ext = X_train.copy(), X_val.copy(), X_test_ext.copy()
    
   
    print("Imputing NaNs and Infs strictly from X_train...")
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(X_train[col]).any() or np.isinf(X_val[col]).any() or np.isinf(X_test_ext[col]).any():
            X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
            X_val[col] = X_val[col].replace([np.inf, -np.inf], np.nan)
            X_test_ext[col] = X_test_ext[col].replace([np.inf, -np.inf], np.nan)
            
        if X_train[col].isna().any() or X_val[col].isna().any() or X_test_ext[col].isna().any():
            median_train_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_train_val)
            X_val[col] = X_val[col].fillna(median_train_val)
            X_test_ext[col] = X_test_ext[col].fillna(median_train_val)
            
    
    # Fit Scaler on the training subset of the training dataset
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform validation and external test set using the same scaling parameters
    X_val_scaled = scaler.transform(X_val)
    X_test_ext_scaled = scaler.transform(X_test_ext)
    
    return X_train_scaled, X_val_scaled, X_test_ext_scaled, y_train, y_val, y_test_ext

def evaluate_model(model, X_test, y_test, name):
    print(f"Evaluating {name}...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    duration = time.time() - start_time
    
    y_probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    
    auc_pr = average_precision_score(y_test, y_probs)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    print(f"Result for {name}: Acc={acc:.4f}, F1={f1:.4f}, AUC-PR={auc_pr:.4f}, Time={duration:.2f}s")
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "far": far,
        "auc_pr": auc_pr,
        "duration": duration,
        "confusion_matrix": cm.tolist(),
        "classification_report": cr
    }

def run_experiment(train_name, train_path, test_name, test_path):
    print(f"\n======================================")
    print(f"Experiment: Train on {train_name} -> Test on {test_name}")
    print(f"======================================")
    
    # 1. Load and Scale
    X_train, X_val, X_test, y_train, y_val, y_test = load_split_scale(train_path, test_path)
    
    # 2. Train Model (Random Forest, Balanced, depth=15)
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # 3. Evaluate on Internal Validation (Same Distribution)
    print("\n--- Internal Validation (Same Distribution) ---")
    res_val = evaluate_model(rf, X_val, y_val, f"{train_name}_Validation")
    
    # 4. Evaluate on External Test (Cross-Dataset)
    print("\n--- External Test (Cross-Dataset) ---")
    res_ext = evaluate_model(rf, X_test, y_test, f"{test_name}_External")
    
    return {
        "experiment": f"{train_name}_to_{test_name}",
        "internal_validation": res_val,
        "external_test": res_ext
    }

def main():
    os.makedirs(METRICS_PATH, exist_ok=True)
    
    cic_path = os.path.join(CROSS_DATA_PATH, "cic_shared_raw.parquet")
    unsw_path = os.path.join(CROSS_DATA_PATH, "unsw_shared_raw.parquet")
    
    if not os.path.exists(cic_path) or not os.path.exists(unsw_path):
         print("Error: Run cross_dataset_utils.py first to generate raw mapped parquets.")
         return
         
    results = []
    
    # Experiment 1: Train CIC and Test UNSW
    res1 = run_experiment("CIC-IDS2017", cic_path, "UNSW-NB15", unsw_path)
    results.append(res1)
    
    # Experiment 2: Train UNSW and Test CIC
    res2 = run_experiment("UNSW-NB15", unsw_path, "CIC-IDS2017", cic_path)
    results.append(res2)
    
    with open(os.path.join(METRICS_PATH, "cross_dataset_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nPhase 7 Complete. Results saved to {METRICS_PATH}/cross_dataset_results.json")

if __name__ == "__main__":
    main()
