import json
import os

def generate_performance_table(json_path):
    """Generates a LaTeX table for binary classification performance."""
    if not os.path.exists(json_path):
        return "Metrics file not found."
        
    with open(json_path, 'r') as f:
        results = json.load(f)
        
    latex_code = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\begin{tabular}{|l|c|c|c|c|c|}",
        "\\hline",
        "Model & Accuracy & Precision & Recall & F1-Score & AUC-PR \\\\",
        "\\hline"
    ]
    
    for res in results:
        auc_pr = f"{res.get('auc_pr', 0):.4f}" if 'auc_pr' in res else "N/A"
        row = f"{res['model']} & {res['accuracy']:.4f} & {res['precision']:.4f} & {res['recall']:.4f} & {res['f1']:.4f} & {auc_pr} \\\\"
        latex_code.append(row)
        latex_code.append("\\hline")
        
    latex_code.extend([
        "\\end{tabular}",
        "\\caption{Binary Classification Performance comparison for CIC-IDS2017 Dataset}",
        "\\label{tab:binary_performance}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_code)

if __name__ == "__main__":
    METRICS_FILE = "artifacts/metrics/binary_results.json"
    print("\n--- GENERATED LATEX TABLE ---\n")
    print(generate_performance_table(METRICS_FILE))
    print("\n----------------------------\n")
