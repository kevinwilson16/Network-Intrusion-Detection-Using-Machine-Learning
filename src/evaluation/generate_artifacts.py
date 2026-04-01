import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for academic plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# Paths
METRICS_PATH = "artifacts/metrics"
MODELS_PATH = "artifacts/models"
PLOTS_PATH = "artifacts/plots"

os.makedirs(PLOTS_PATH, exist_ok=True)

# ---------------------------------------------------------
# LaTeX Table Generators
# ---------------------------------------------------------

def generate_binary_latex():
    print("\n% ==========================================================")
    print("% Table X: Binary Baseline Comparison (CIC-IDS2017 vs UNSW-NB15)")
    print("% ==========================================================\n")
    
    with open(os.path.join(METRICS_PATH, "binary_results.json"), 'r') as f:
        cic_data = json.load(f)
    with open(os.path.join(METRICS_PATH, "unsw_binary_results.json"), 'r') as f:
        unsw_data = json.load(f)
        
    print('''\\begin{table}[h!]
\\centering
\\begin{tabular}{|l|c|c|c|c|c|c|}
\\hline
\\textbf{Model} & \\multicolumn{3}{c|}{\\textbf{CIC-IDS2017}} & \\multicolumn{3}{c|}{\\textbf{UNSW-NB15}} \\\\
\\cline{2-7}
& Acc & F1 & AUC-PR & Acc & F1 & AUC-PR \\\\
\\hline''')
    
    for c_model, u_model in zip(cic_data, unsw_data):
        # We assume order is LogisticRegression, RandomForest
        model_name = c_model.get('model', c_model.get('name', 'Unknown'))
        print(f"{model_name} & "
              f"{c_model['accuracy']:.4f} & {c_model['f1']:.4f} & {c_model.get('auc_pr', 0):.4f} & "
              f"{u_model['accuracy']:.4f} & {u_model['f1']:.4f} & {u_model.get('auc_pr', 0):.4f} \\\\")
              
    print('''\\hline
\\end{tabular}
\\caption{Binary Baseline Performance across Datasets (Class Weight: Balanced)}
\\label{tab:binary_comparison}
\\end{table}''')

def generate_cross_dataset_latex():
    print("\n% ==========================================================")
    print("% Table Y: Cross-Dataset Generalization Deterioration")
    print("% ==========================================================\n")
    
    with open(os.path.join(METRICS_PATH, "cross_dataset_results.json"), 'r') as f:
        data = json.load(f)
        
    print('''\\begin{table}[h!]
\\centering
\\begin{tabular}{|l|c|c|c|c|}
\\hline
\\textbf{Experiment} & \\multicolumn{2}{c|}{\\textbf{Internal Validation}} & \\multicolumn{2}{c|}{\\textbf{External Test}} \\\\
\\cline{2-5}
& Acc & F1 & Acc & F1 \\\\
\\hline''')
    
    for exp in data:
        name = exp['experiment'].replace('_to_', ' $\\rightarrow$ ')
        internal = exp['internal_validation']
        external = exp['external_test']
        
        print(f"{name} & "
              f"{internal['accuracy']:.4f} & {internal['f1']:.4f} & "
              f"{external['accuracy']:.4f} & {external['f1']:.4f} \\\\")
              
    print('''\\hline
\\end{tabular}
\\caption{Generalization Deterioration: Internal vs External Evaluation (9 Shared Features)}
\\label{tab:generalization_cross}
\\end{table}''')

# ---------------------------------------------------------
# Visualization Generators
# ---------------------------------------------------------

def plot_feature_importance():
    print("Generating Feature Importance Plots...")
    
    # CIC-IDS2017
    try:
        model_cic = joblib.load(os.path.join(MODELS_PATH, "xgb_multiclass_balanced.pkl"))
        # We need feature names, let's load a sample of processed data
        sample_cic = pd.read_parquet(f"data/cicids2017/processed/train_multiclass.parquet")
        cic_features = sample_cic.drop(columns=['multiclass_label', 'label', 'is_attack'], errors='ignore').columns
        
        importances = model_cic.feature_importances_
        indices = np.argsort(importances)[::-1][:15] # Top 15
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[cic_features[i] for i in indices], palette="viridis")
        plt.title('Top 15 Feature Importances (XGBoost) - CIC-IDS2017')
        plt.xlabel('Relative Importance (Gini)')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_PATH, "feature_importance_rf_cic.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Skipping CIC Feature Importance: {e}")

    # UNSW-NB15
    try:
        model_unsw = joblib.load(os.path.join(MODELS_PATH, "xgb_unsw_balanced.pkl"))
        sample_unsw = pd.read_parquet(f"data/unsw-nb15/processed/train_unsw.parquet")
        unsw_features = sample_unsw.drop(columns=['multiclass_label', 'label'], errors='ignore').columns
        
        importances = model_unsw.feature_importances_
        # Ensure we don't index out of bounds if there are fewer than 15 features
        num_features = min(15, len(importances))
        indices = np.argsort(importances)[::-1][:num_features]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[unsw_features[i] for i in indices], palette="magma")
        plt.title('Top Feature Importances (XGBoost) - UNSW-NB15')
        plt.xlabel('Relative Importance (Gini)')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_PATH, "feature_importance_rf_unsw.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Skipping UNSW Feature Importance: {e}")

def plot_hybrid_bottleneck():
    print("Generating Hybrid Bottleneck Comparison Plots...")
    
    # CIC-IDS2017
    try:
        with open(os.path.join(METRICS_PATH, "multiclass_results.json"), 'r') as f:
            cic_multi = json.load(f)
            # Index 1 is XGB SMOTE
            xgb_cic = cic_multi['results'][1]['report']
            
        with open(os.path.join(METRICS_PATH, "hybrid_results.json"), 'r') as f:
            hyb_cic = json.load(f)['report']
            
        classes = [c for c in xgb_cic.keys() if c not in ['accuracy', 'macro avg', 'weighted avg', 'BENIGN'] and c != 'NaN']
        # Filter to top 7 most affected/interesting classes
        classes = sorted(classes)[:7]
        
        xgb_f1s = [xgb_cic[c]['f1-score'] for c in classes]
        hyb_f1s = [hyb_cic.get(c, {}).get('f1-score', 0) for c in classes]
        
        df_cic = pd.DataFrame({
            'Attack Class': classes * 2,
            'F1-Score': xgb_f1s + hyb_f1s,
            'Model': ['Standalone XGB (SMOTE)'] * len(classes) + ['Hybrid (IF -> XGB)'] * len(classes)
        })
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_cic, x='Attack Class', y='F1-Score', hue='Model', palette="Set2")
        plt.title('The Bottleneck Effect: Standalone vs Hybrid F1-Scores (CIC-IDS2017)')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_PATH, "f1_comparison_hybrid_cic.png"), dpi=300)
        plt.close()
    except Exception as e:
         print(f"Skipping CIC Hybrid Plot: {e}")

    # UNSW-NB15
    try:
        with open(os.path.join(METRICS_PATH, "unsw_multiclass_results.json"), 'r') as f:
             multi_data = json.load(f)
             # XGB SMOTE is index 1 in supervised
             xgb_unsw = multi_data['supervised'][1]['report']
             
        with open(os.path.join(METRICS_PATH, "unsw_hybrid_results.json"), 'r') as f:
             hyb_data = json.load(f)
             hyb_unsw = hyb_data['report']
             
        classes = [c for c in xgb_unsw.keys() if c not in ['accuracy', 'macro avg', 'weighted avg', 'Normal']]
        classes = sorted(classes)[:7]
        
        xgb_f1s = [xgb_unsw[c]['f1-score'] for c in classes]
        hyb_f1s = [hyb_unsw.get(c, {}).get('f1-score', 0) for c in classes]
        
        df_unsw = pd.DataFrame({
            'Attack Class': classes * 2,
            'F1-Score': xgb_f1s + hyb_f1s,
            'Model': ['Standalone XGB (SMOTE)'] * len(classes) + ['Hybrid (IF -> XGB)'] * len(classes)
        })
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_unsw, x='Attack Class', y='F1-Score', hue='Model', palette="Set1")
        plt.title('The Extreme Bottleneck Effect: Standalone vs Hybrid F1-Scores (UNSW-NB15)')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_PATH, "f1_comparison_hybrid_unsw.png"), dpi=300)
        plt.close()
    except Exception as e:
         print(f"Skipping UNSW Hybrid Plot: {e}")

def plot_generalization():
    print("Generating Cross-Dataset Generalization Plot...")
    try:
        with open(os.path.join(METRICS_PATH, "cross_dataset_results.json"), 'r') as f:
            data = json.load(f)
            
        experiments = [exp['experiment'].replace('_to_', ' -> ') for exp in data]
        internal_f1s = [exp['internal_validation']['f1'] for exp in data]
        external_f1s = [exp['external_test']['f1'] for exp in data]
        
        df = pd.DataFrame({
            'Experiment': experiments * 2,
            'F1-Score': internal_f1s + external_f1s,
            'Evaluation Set': ['Internal Validation'] * 2 + ['External Test'] * 2
        })
        
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df, x='Experiment', y='F1-Score', hue='Evaluation Set', palette="mako")
        plt.title('Generalization Deterioration across Datasets (F1-Score)')
        plt.ylim(0, 1.1)
        for i, p in enumerate(plt.gca().patches):
             height = p.get_height()
             if height > 0:
                 plt.gca().text(p.get_x() + p.get_width()/2., height + 0.02, '{:1.4f}'.format(height), ha="center")
             elif height == 0:
                 plt.gca().text(p.get_x() + p.get_width()/2., height + 0.02, '0.0000', ha="center", color='red', fontweight='bold')
             
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_PATH, "generalization_deterioration.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Skipping Generalization Plot: {e}")

if __name__ == "__main__":
    print("--- GENERATING DISSERTATION ARTIFACTS ---")
    generate_binary_latex()
    generate_cross_dataset_latex()
    
    plot_feature_importance()
    plot_hybrid_bottleneck()
    plot_generalization()
    
    print("\nArtifact Generation Complete. Plots saved to artifacts/plots/")
