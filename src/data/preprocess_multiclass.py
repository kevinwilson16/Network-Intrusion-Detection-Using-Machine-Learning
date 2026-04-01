import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
import pyarrow as pa
import pyarrow.parquet as pq

# Constants for Paths
RAW_DATA_PATH = "data/cicids2017/raw"
PROCESSED_DATA_PATH = "data/cicids2017/processed"

def load_and_concat_raw_data(data_path):
    print(f"Loading raw data from {data_path}...")
    import glob
    all_files = glob.glob(os.path.join(data_path, "*.csv"))
    df_list = []
    for file in all_files:
        print(f"Reading {os.path.basename(file)}...")
        df = pd.read_csv(file, skipinitialspace=True, encoding='latin1')
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def clean_data(df):
    print("Cleaning dataset...")
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(' ', '_')
    
    # Handle Infinity: Replace with NaN to be imputed after split
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def main():
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    
    df = load_and_concat_raw_data(RAW_DATA_PATH)
    df = clean_data(df)
    
    print("Encoding labels for multiclass...")
    le = LabelEncoder()
    df['multiclass_label'] = le.fit_transform(df['label'])
    
    # Save the label encoder for inverse mapping during evaluation
    os.makedirs("artifacts/models", exist_ok=True)
    joblib.dump(le, "artifacts/models/multiclass_label_encoder.pkl")
    
    # Features and Target
    X = df.drop(columns=['label', 'multiclass_label'])
    y = df['multiclass_label']
    original_str_labels = df['label']
    
    # Drop leakage features (IPs, Ports, Timestamps)
    cols_to_drop = ['flow_id', 'source_ip', 'destination_ip', 'timestamp', 'source_port', 'destination_port']
    X = X.drop(columns=cols_to_drop, errors='ignore')
    
    # Ensure numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    
    # Stratified Split (80/20)
    print("Splitting data...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    str_labels_train = original_str_labels.iloc[train_idx]
    str_labels_test = original_str_labels.iloc[test_idx]
    
    print("Imputing NaNs based on X_train statistics...")
    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)
    
    # Scaling
    print("Scaling features...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Save to Parquet
    print("Saving multiclass parquets...")
    train_df = pd.concat([X_train_scaled.reset_index(drop=True), y_train.reset_index(drop=True), str_labels_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_scaled.reset_index(drop=True), y_test.reset_index(drop=True), str_labels_test.reset_index(drop=True)], axis=1)
    
    train_df.to_parquet(os.path.join(PROCESSED_DATA_PATH, "train_multiclass.parquet"), index=False)
    test_df.to_parquet(os.path.join(PROCESSED_DATA_PATH, "test_multiclass.parquet"), index=False)
    
    print(f"Multiclass preprocessing complete. Classes found: {le.classes_}")

if __name__ == "__main__":
    main()
