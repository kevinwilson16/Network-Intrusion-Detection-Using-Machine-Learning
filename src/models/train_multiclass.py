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
    
    X_train, y_train, X_test, y_test = load_data()
    le = joblib.load(os.path.join(MODELS_PATH, "multiclass_label_encoder.pkl"))
    target_names = le.classes_
    
    results = []
    
    # 1. Random Forest with Balanced Class Weights
    rf_balanced = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42, class_weight='balanced')
    res_rf_bal = train_and_eval("RF_BalancedWeights", rf_balanced, X_train, y_train, X_test, y_test, target_names)
    results.append(res_rf_bal)
    joblib.dump(rf_balanced, os.path.join(MODELS_PATH, "rf_multiclass_balanced.pkl"))
    
    # 2. XGBoost with SMOTE (Oversampling)
    # SMOTE on 2M+ rows (with 29M resampled rows) is too slow for local run.
    # Academic compromise: Downsample the majority class (BENIGN) to 500k, then apply SMOTE.
    print("Downsampling majority class (BENIGN) for SMOTE feasibility...")
    
    # Identify the majority class (usually index 0 but we check carefully)
    counts = y_train.value_counts()
    majority_class_idx = counts.idxmax()
    
    # Separate majority and minority
    mask_majority = (y_train == majority_class_idx)
    X_train_maj = X_train[mask_majority]
    y_train_maj = y_train[mask_majority]
    
    X_train_min = X_train[~mask_majority]
    y_train_min = y_train[~mask_majority]
    
    # Downsample majority to 100k
    X_maj_down = X_train_maj.sample(n=100000, random_state=42)
    y_maj_down = y_train_maj.loc[X_maj_down.index]
    
    X_train_eff = pd.concat([X_maj_down, X_train_min])
    y_train_eff = pd.concat([y_maj_down, y_train_min])
    
    print(f"Applying SMOTE on efficient dataset (Size: {len(X_train_eff)})...")
    smote = SMOTE(random_state=42, k_neighbors=1)
    
    try:
        X_res, y_res = smote.fit_resample(X_train_eff, y_train_eff)
        print(f"SMOTE Counts: {len(X_train_eff)} -> {len(X_res)}")
        
        xgb_smote = XGBClassifier(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42, tree_method='hist')
        res_xgb_smote = train_and_eval("XGB_SMOTE", xgb_smote, X_res, y_res, X_test, y_test, target_names)
        results.append(res_xgb_smote)
        joblib.dump(xgb_smote, os.path.join(MODELS_PATH, "xgb_multiclass_smote.pkl"))
    except Exception as e:
        print(f"SMOTE experiment failed: {e}")

    
    # Save Results
    with open(os.path.join(METRICS_PATH, "multiclass_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    print("Phase 3 Training Complete.")

if __name__ == "__main__":
    main()
