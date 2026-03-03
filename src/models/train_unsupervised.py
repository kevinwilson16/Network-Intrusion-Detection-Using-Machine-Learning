import os
import joblib
import pandas as pd
import numpy as np
import time
import json
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# Paths
PROCESSED_DATA_PATH = "data/cicids2017/processed"
MODELS_PATH = "artifacts/models"
METRICS_PATH = "artifacts/metrics"

def load_data():
    print("Loading data for unsupervised training...")
    # We use the multiclass parquet because it contains the full set with original labels
    train_df = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "train_multiclass.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "test_multiclass.parquet"))
    
    X_train = train_df.drop(columns=['multiclass_label', 'label'])
    # Convert string labels to binary for evaluation (BENIGN = 0, Attack = 1)
    y_train_binary = (train_df['label'] != 'BENIGN').astype(int)
    
    X_test = test_df.drop(columns=['multiclass_label', 'label'])
    y_test_binary = (test_df['label'] != 'BENIGN').astype(int)
    
    return X_train, y_train_binary, X_test, y_test_binary

def main():
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(METRICS_PATH, exist_ok=True)
    
    X_train, y_train, X_test, y_test = load_data()
    
    # Calculate natural attack frequency for reference
    natural_freq = y_train.mean()
    print(f"Dataset Info: Training Size={len(X_train)} | Natural Attack Frequency={natural_freq:.4f}")

    results = []
    contamination_levels = [0.10, 0.15, 0.20]

    for contam in contamination_levels:
        print(f"\n--- Experiment: Isolation Forest (Contamination={contam}) ---")
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=contam,
            random_state=42,
            n_jobs=-1
        )
        
        start_time = time.time()
        # True Unsupervised: Labels are NOT passed to fit
        iso_forest.fit(X_train)
        duration = time.time() - start_time
        
        # Prediction: 1 = Normal, -1 = Anomaly
        preds_raw = iso_forest.predict(X_test)
        
        # Map to our metrics: 1 = Attack, 0 = Normal
        # Logic: If -1 (Anomaly), then 1 (Attack). If 1 (Normal), then 0 (Normal).
        y_pred = np.where(preds_raw == -1, 1, 0)
        
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        
        print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        results.append({
            "contamination": contam,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "duration": duration
        })
        
        # Save the 0.15 model as the primary unsupervised model artifact
        if contam == 0.15:
            joblib.dump(iso_forest, os.path.join(MODELS_PATH, "isolation_forest_base.pkl"))

    # Save Results
    output = {
        "experiment_name": "Isolation Forest Sensitivity Analysis",
        "natural_attack_frequency": natural_freq,
        "results": results,
        "mapping_logic": "-1 (Anomaly) -> 1 (Attack), 1 (Normal) -> 0 (Normal)"
    }
    
    with open(os.path.join(METRICS_PATH, "unsupervised_results.json"), "w") as f:
        json.dump(output, f, indent=4)
        
    print(f"\nPhase 4 Unsupervised Training Complete. Results saved to {METRICS_PATH}")

if __name__ == "__main__":
    main()
