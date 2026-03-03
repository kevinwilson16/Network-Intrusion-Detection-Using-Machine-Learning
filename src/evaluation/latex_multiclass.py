import json
import os
import pandas as pd

def generate_multiclass_table(json_path):
    """Generates a LaTeX table showing per-class F1-scores for different techniques."""
    if not os.path.exists(json_path):
        return "Metrics file not found."
        
    with open(json_path, 'r') as f:
        results = json.load(f)
        
    # We want to compare F1-scores for each class across models
    classes = [c for c in results[0]['report'].keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
    
    latex_code = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\begin{tabular}{|l|c|c|}",
        "\\hline",
        "Attack Category & RF (Balanced Weights) & XGB (SMOTE) \\\\",
        "\\hline"
    ]
    
    for cls in classes:
        f1_rf = results[0]['report'][cls]['f1-score']
        f1_xgb = results[1]['report'][cls]['f1-score'] if len(results) > 1 else 0.0
        row = f"{cls.replace('_', ' ')} & {f1_rf:.4f} & {f1_xgb:.4f} \\\\"
        latex_code.append(row)
        latex_code.append("\\hline")
        
    # Add Overall Metrics
    latex_code.append("Overall Accuracy & {:.4f} & {:.4f} \\\\".format(
        results[0]['accuracy'],
        results[1]['accuracy'] if len(results) > 1 else 0.0
    ))
    latex_code.append("\\hline")
    
    latex_code.extend([
        "\\end{tabular}",
        "\\caption{Multiclass Performance: Impact of Imbalance Handling on F1-Scores}",
        "\\label{tab:multiclass_imbalance}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_code)

if __name__ == "__main__":
    METRICS_FILE = "artifacts/metrics/multiclass_results.json"
    print("\n--- GENERATED LATEX MULTICLASS TABLE ---\n")
    print(generate_multiclass_table(METRICS_FILE))
    print("\n---------------------------------------\n")
