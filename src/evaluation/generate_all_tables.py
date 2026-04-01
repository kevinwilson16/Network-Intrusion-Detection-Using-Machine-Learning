import os
import json

# Navigate to the metrics directory relative to this script
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
METRICS_DIR = os.path.join(BASE_DIR, "artifacts", "metrics")

def load_json(filename):
    filepath = os.path.join(METRICS_DIR, filename)
    with open(filepath, 'r') as f:
        return json.load(f)

def print_binary_table():
    cic_data = load_json("binary_results.json")
    unsw_data = load_json("unsw_binary_results.json")
    
    cic_lr = next(item for item in cic_data if "Logistic" in item.get('model', item.get('name', '')))
    cic_rf = next(item for item in cic_data if "Random Forest" in item.get('model', item.get('name', '')))
    
    unsw_lr = next(item for item in unsw_data if "Logistic" in item.get('name', item.get('model', '')))
    unsw_rf = next(item for item in unsw_data if "RandomForest" in item.get('name', item.get('model', '')))
    
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Comparison of Supervised Binary Baselines: Logistic Regression vs. Random Forest}")
    print(r"  \label{tab:binary_baselines}")
    print(r"  \begin{tabular}{llcccc}")
    print(r"    \toprule")
    print(r"    \textbf{Dataset} & \textbf{Model} & \textbf{Accuracy} & \textbf{F1-Score} & \textbf{AUC-PR} & \textbf{Time (s)} \\")
    print(r"    \midrule")
    
    print(rf"    CIC-IDS2017 & Logistic Regression & {cic_lr['accuracy']:.4f} & {cic_lr['f1']:.4f} & {cic_lr['auc_pr']:.4f} & {cic_lr['duration']:.2f} \\")
    print(rf"                & Random Forest & {cic_rf['accuracy']:.4f} & {cic_rf['f1']:.4f} & {cic_rf['auc_pr']:.4f} & {cic_rf['duration']:.2f} \\")
    print(r"    \midrule")
    print(rf"    UNSW-NB15   & Logistic Regression & {unsw_lr['accuracy']:.4f} & {unsw_lr['f1']:.4f} & {unsw_lr['auc_pr']:.4f} & {unsw_lr['duration']:.2f} \\")
    print(rf"                & Random Forest & {unsw_rf['accuracy']:.4f} & {unsw_rf['f1']:.4f} & {unsw_rf['auc_pr']:.4f} & {unsw_rf['duration']:.2f} \\")
    
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    print("\n")

def print_multiclass_cic_table():
    data = load_json("multiclass_results.json")['results']
    bal = next(x for x in data if "BalancedWeights" in x["name"])
    smote = next(x for x in data if "SMOTE" in x["name"])
    
    classes = [k for k in bal['report'].keys() if k not in ['accuracy', 'macro avg', 'weighted avg', 'NaN']]
    
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Multiclass Performance (CIC-IDS2017): XGBoost with Balanced Weights vs. SMOTE}")
    print(r"  \label{tab:multiclass_cic}")
    print(r"  \begin{tabular}{lcc}")
    print(r"    \toprule")
    print(r"    \textbf{Attack Category} & \textbf{XGBoost (Balanced Weights) F1} & \textbf{XGBoost (SMOTE) F1} \\")
    print(r"    \midrule")
    
    for c in sorted(classes):
        f1_bal = bal['report'].get(c, {}).get('f1-score', 0.0)
        f1_smote = smote['report'].get(c, {}).get('f1-score', 0.0)
        c_clean = c.replace('_', r'\_').replace('&', r'\&').replace('\u0096', '-')
        print(rf"    {c_clean} & {f1_bal:.4f} & {f1_smote:.4f} \\")
        
    print(r"    \midrule")
    macro_bal = bal['report']['macro avg']['f1-score']
    macro_smote = smote['report']['macro avg']['f1-score']
    print(rf"    \textbf{{Macro Average}} & \textbf{{{macro_bal:.4f}}} & \textbf{{{macro_smote:.4f}}} \\")
    
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    print("\n")

def print_multiclass_unsw_table():
    data = load_json("unsw_multiclass_results.json")['supervised']
    bal = next(x for x in data if "BalancedWeights" in x["name"])
    smote = next(x for x in data if "SMOTE" in x["name"])
    
    classes = [k for k in bal['report'].keys() if k not in ['accuracy', 'macro avg', 'weighted avg', 'NaN']]
    
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Multiclass Performance (UNSW-NB15): XGBoost with Balanced Weights vs. SMOTE}")
    print(r"  \label{tab:multiclass_unsw}")
    print(r"  \begin{tabular}{lcc}")
    print(r"    \toprule")
    print(r"    \textbf{Attack Category} & \textbf{XGBoost (Balanced Weights) F1} & \textbf{XGBoost (SMOTE) F1} \\")
    print(r"    \midrule")
    
    for c in sorted(classes):
        f1_bal = bal['report'].get(c, {}).get('f1-score', 0.0)
        f1_smote = smote['report'].get(c, {}).get('f1-score', 0.0)
        c_clean = c.replace('_', r'\_').replace('&', r'\&').replace('\u0096', '-')
        print(rf"    {c_clean} & {f1_bal:.4f} & {f1_smote:.4f} \\")
        
    print(r"    \midrule")
    macro_bal = bal['report']['macro avg']['f1-score']
    macro_smote = smote['report']['macro avg']['f1-score']
    print(rf"    \textbf{{Macro Average}} & \textbf{{{macro_bal:.4f}}} & \textbf{{{macro_smote:.4f}}} \\")
    
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    print("\n")

def print_hybrid_table():
    cic_multi = load_json("multiclass_results.json")['results']
    cic_xgb_smote = next(x for x in cic_multi if "SMOTE" in x["name"])['report']['macro avg']['f1-score']
    
    unsw_multi = load_json("unsw_multiclass_results.json")['supervised']
    unsw_xgb_smote = next(x for x in unsw_multi if "SMOTE" in x["name"])['report']['macro avg']['f1-score']
    
    cic_hybrid = load_json("hybrid_results.json")['report']['macro avg']['f1-score']
    unsw_hybrid = load_json("unsw_hybrid_results.json")['report']['macro avg']['f1-score']
    
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{The Bottleneck Effect: Standalone Supervised XGBoost vs. Two-Stage Hybrid Pipeline}")
    print(r"  \label{tab:hybrid_bottleneck}")
    print(r"  \begin{tabular}{lcc}")
    print(r"    \toprule")
    print(r"    \textbf{Dataset} & \textbf{Standalone XGBoost (SMOTE) F1} & \textbf{Two-Stage Hybrid F1} \\")
    print(r"    \midrule")
    print(rf"    CIC-IDS2017 & {cic_xgb_smote:.4f} & {cic_hybrid:.4f} \\")
    print(rf"    UNSW-NB15 & {unsw_xgb_smote:.4f} & {unsw_hybrid:.4f} \\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    print("\n")

def print_cross_dataset_table():
    data = load_json("cross_dataset_results.json")
    
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Topological Overfitting: Internal Validation vs. External Test Generalisation}")
    print(r"  \label{tab:cross_dataset}")
    print(r"  \begin{tabular}{llccc}")
    print(r"    \toprule")
    print(r"    \textbf{Experiment Setup} & \textbf{Evaluation Phase} & \textbf{Accuracy} & \textbf{F1-Score} & \textbf{AUC-PR} \\")
    print(r"    \midrule")
    
    for idx, row in enumerate(data):
        exp_name = row['experiment'].replace('_to_', r' $\rightarrow$ ').replace('_', r'\_')
        int_val = row['internal_validation']
        ext_test = row['external_test']
        
        # Note: requires \usepackage{multirow} in LaTeX preamble
        print(rf"    \multirow{{2}}{{*}}{{{exp_name}}} & Internal Validation & {int_val['accuracy']:.4f} & {int_val['f1']:.4f} & {int_val['auc_pr']:.4f} \\")
        print(rf"    & External Test & {ext_test['accuracy']:.4f} & {ext_test['f1']:.4f} & {ext_test['auc_pr']:.4f} \\")
        if idx < len(data) - 1:
            print(r"    \midrule")
            
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    print("\n")

if __name__ == '__main__':
    print("%%% GENERATED LATEX TABLES %%%\n")
    print_binary_table()
    print_multiclass_cic_table()
    print_multiclass_unsw_table()
    print_hybrid_table()
    print_cross_dataset_table()
