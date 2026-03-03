import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import time

# Paths
PROCESSED_DATA_PATH = "data/cicids2017/processed"
MODELS_PATH = "artifacts/models"

def load_data():
    print("Loading processed training data...")
    train_df = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "train.parquet"))
    # The last two columns were 'is_attack' and 'label' (original string)
    # We drop both for training features
    X_train = train_df.drop(columns=['is_attack', 'label'])
    y_train = train_df['is_attack']
    return X_train, y_train

def train_logistic_regression(X_train, y_train):
    print("Training Logistic Regression...")
    start_time = time.time()
    lr = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42, class_weight='balanced')
    lr.fit(X_train, y_train)
    duration = time.time() - start_time
    print(f"Logistic Regression trained in {duration:.2f}s")
    return lr

def train_random_forest(X_train, y_train):
    print("Training Random Forest...")
    start_time = time.time()
    # Using smaller n_estimators for MVP speed, max_depth to prevent overfitting
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    duration = time.time() - start_time
    print(f"Random Forest trained in {duration:.2f}s")
    return rf

def main():
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    X_train, y_train = load_data()
    
    # Train Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    joblib.dump(lr_model, os.path.join(MODELS_PATH, "lr_binary.pkl"))
    
    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train)
    joblib.dump(rf_model, os.path.join(MODELS_PATH, "rf_binary.pkl"))
    
    print(f"Models saved to {MODELS_PATH}")

if __name__ == "__main__":
    main()
