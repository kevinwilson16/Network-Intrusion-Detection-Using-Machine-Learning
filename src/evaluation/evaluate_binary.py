import os
import joblib
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_curve, auc
)

# Paths
PROCESSED_DATA_PATH = "data/cicids2017/processed"
MODELS_PATH = "artifacts/models"
METRICS_PATH = "artifacts/metrics"
PLOTS_PATH = "artifacts/plots"

def load_test_data():
    print("Loading processed test data...")
    test_df = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "test.parquet"))
    X_test = test_df.drop(columns=['is_attack', 'label'])
    y_test = test_df['is_attack']
    return X_test, y_test

def evaluate_model(model, X_test, y_test, model_name):
    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    
    if y_prob is not None:
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        metrics["auc_pr"] = auc(recall, precision)
        
    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(PLOTS_PATH, f"cm_{model_name.lower().replace(' ', '_')}.png"))
    plt.close()
    
    return metrics

def main():
    os.makedirs(METRICS_PATH, exist_ok=True)
    os.makedirs(PLOTS_PATH, exist_ok=True)
    
    X_test, y_test = load_test_data()
    
    all_results = []
    
    model_files = {
        "Logistic Regression": "lr_binary.pkl",
        "Random Forest": "rf_binary.pkl"
    }
    
    for name, filename in model_files.items():
        model_path = os.path.join(MODELS_PATH, filename)
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            metrics = evaluate_model(model, X_test, y_test, name)
            all_results.append(metrics)
        else:
            print(f"Warning: Model file {filename} not found.")
            
    # Save as JSON
    with open(os.path.join(METRICS_PATH, "binary_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"Results saved to {METRICS_PATH}")

if __name__ == "__main__":
    main()
