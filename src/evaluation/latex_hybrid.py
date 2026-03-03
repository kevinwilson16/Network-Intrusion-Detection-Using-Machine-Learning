import os
import json

def generate_hybrid_table(multiclass_json, hybrid_json):
    """Generates a LaTeX table comparing Standalone XGBoost vs Hybrid Pipeline F1-scores."""
    if not os.path.exists(multiclass_json) or not os.path.exists(hybrid_json):
        return "Metrics files not found."
        
    with open(multiclass_json, 'r') as f:
        data_multi = json.load(f)
        # XGBoost SMOTE is at index 1
        results_xgb = data_multi['results'][1]['report']
        acc_xgb = data_multi['results'][1]['accuracy']
        
    with open(hybrid_json, 'r') as f:
        data_hybrid = json.load(f)
        results_hyb = data_hybrid['report']
        acc_hyb = data_hybrid['accuracy']
        
    classes = [c for c in results_xgb.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
    
    latex_code = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\begin{tabular}{|l|c|c|}",
        "\\hline",
        "Attack Category & Standalone XGB (SMOTE) & Hybrid (IF $\\rightarrow$ XGB) \\\\",
        "\\hline"
    ]
    
    for cls in classes:
        f1_xgb = results_xgb[cls]['f1-score']
        # Hybrid might miss some classes if IF didn't flag them
        f1_hyb = results_hyb.get(cls, {}).get('f1-score', 0.0)
        row = f"{cls.replace('_', ' ')} & {f1_xgb:.4f} & {f1_hyb:.4f} \\\\"
        latex_code.append(row)
        latex_code.append("\\hline")
        
    # Add Overall Metrics
    latex_code.append("Overall Accuracy & {:.4f} & {:.4f} \\\\".format(
        acc_xgb,
        acc_hyb
    ))
    latex_code.append("\\hline")
    
    latex_code.extend([
        "\\end{tabular}",
        "\\caption{Performance Comparison: Standalone XGBoost vs. Two-Stage Hybrid Pipeline}",
        "\\label{tab:hybrid_comparison}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_code)

if __name__ == "__main__":
    MULTI_FILE = "artifacts/metrics/multiclass_results.json"
    HYBRID_FILE = "artifacts/metrics/hybrid_results.json"
    print("\n--- GENERATED LATEX HYBRID COMPARISON TABLE ---\n")
    print(generate_hybrid_table(MULTI_FILE, HYBRID_FILE))
