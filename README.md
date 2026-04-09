# Machine Learning-Driven Anomaly Detection in Network Traffic for Enhanced Cyber Security

This project evaluates supervised, unsupervised, and hybrid machine learning models to detect malicious network traffic. A key focus is placed on handling severe class imbalance and mitigating topological overfitting to create robust intrusion detection systems.

## Datasets Used

- **CIC-IDS2017**
- **UNSW-NB15**

## Models Implemented

- Supervised Binary (Logistic Regression, Random Forest)
- Supervised Multiclass (XGBoost with SMOTE and Balanced Weights)
- Unsupervised Anomaly Detection (Isolation Forest)
- Two-Stage Hybrid Pipeline (Isolation Forest $\rightarrow$ XGBoost)

## Repository Structure

```text
.
├── src/
│   ├── data/          # Data preprocessing, cleaning, and dataset splitting
│   ├── models/        # Model architectures and training scripts
│   └── evaluation/    # Metric extraction, evaluation pipelines, and visualization
├── requirements.txt   # Python dependencies
```
