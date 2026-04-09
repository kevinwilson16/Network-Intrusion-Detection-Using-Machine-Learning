import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
import pyarrow as pa
import pyarrow.parquet as pq

# Constants for Paths
RAW_DATA_PATH = "data/cicids2017/raw"
PROCESSED_DATA_PATH = "data/cicids2017/processed"

def load_and_concat_raw_data(data_path):
    """Loads all CSVs from the raw directory and concatenates them."""
    print(f"Loading raw data from {data_path}...")
    all_files = glob.glob(os.path.join(data_path, "*.csv"))
    
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}. Please check the directory.")
        
    df_list = []
    total_files = len(all_files)
    for i, file in enumerate(all_files, 1):
        print(f"[{i}/{total_files}] Reading {os.path.basename(file)}...")
        df = pd.read_csv(file, skipinitialspace=True, encoding='latin1')
        df_list.append(df)
        
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Successfully loaded {len(combined_df)} rows and {len(combined_df.columns)} columns.")
    return combined_df

def clean_data(df):
    """Cleans column names, handles infinity/NaNs, and drops bad rows."""
    print("Cleaning dataset...")
    
    # Standardise column names: strip spaces, lowercase, replace spaces with underscores
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(' ', '_')
    

    df['label'] = df['label'].astype(str).str.strip()
    df.loc[df['label'].str.contains('BENIGN', case=False, na=False), 'label'] = 'BENIGN'
    
    return df

def map_binary_labels(df):
    """Maps multiclass labels to binary (0 = Normal, 1 = Attack)."""
    print("Mapping labels to binary...")
    if 'label' not in df.columns:
        raise ValueError("Could not find 'label' column. Columns present: " + str(df.columns))
        
    benign_label = 'BENIGN'
    
    df['is_attack'] = (df['label'] != benign_label).astype(int)
    
    # Log the class distribution
    counts = df['is_attack'].value_counts()
    print(f"Binary Class Distribution:\n0 (Normal): {counts.get(0, 0)}\n1 (Attack): {counts.get(1, 0)}")
    return df


def split_and_scale(df):
    """Performs stratified train/test split and robust scaling to avoid data leakage."""
    print("Splitting and scaling data (80/20 train/test split)...")
    
    X = df.drop(columns=['label', 'is_attack'])
    y = df['is_attack']
    original_labels = df['label']
    
    # Drop leaky features & coerce to numeric
    cols_to_drop = ['flow_id', 'source_ip', 'destination_ip', 'timestamp', 'source_port', 'destination_port']
    X = X.drop(columns=cols_to_drop, errors='ignore')
    for col in X.columns:
         X[col] = pd.to_numeric(X[col], errors='coerce')
         
    print("Replacing Infs with NaNs before the split...")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    labels_train, labels_test = original_labels.iloc[train_idx].copy(), original_labels.iloc[test_idx].copy()
    
    print("Imputing NaNs purely based on X_train statistics...")
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if X_train[col].isna().any() or X_test[col].isna().any():
            median_train_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_train_val)
            X_test[col] = X_test[col].fillna(median_train_val)

    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    train_df = pd.concat([X_train_scaled.reset_index(drop=True), y_train.reset_index(drop=True), labels_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_scaled.reset_index(drop=True), y_test.reset_index(drop=True), labels_test.reset_index(drop=True)], axis=1)
    
    return train_df, test_df

def main():
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    
    df = load_and_concat_raw_data(RAW_DATA_PATH)
    df = clean_data(df)
    df = map_binary_labels(df)
    train_df, test_df = split_and_scale(df)
    
    print("Saving processed files to parquet format...")
    train_path = os.path.join(PROCESSED_DATA_PATH, "train.parquet")
    test_path = os.path.join(PROCESSED_DATA_PATH, "test.parquet")
    
    train_df.to_parquet(train_path, engine='pyarrow', index=False)
    test_df.to_parquet(test_path, engine='pyarrow', index=False)
    
    print(f"Data preprocessing complete! Saved {len(train_df)} train rows and {len(test_df)} test rows.")

if __name__ == "__main__":
    main()
