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
        # CICIDS2017 files can have mixed types or bad rows, we read carefully
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
    
    # Handle Infinity: Cap to max finite value per column
    for col in df.select_dtypes(include=[np.number]).columns:
        max_val = df.loc[df[col] != np.inf, col].max()
        df[col] = df[col].replace(np.inf, max_val)
        
    # Handle NaNs: Impute with Median
    df = df.fillna(df.median(numeric_only=True))
    
    print("Capped Infinity values and imputed NaNs with Median.")
    return df

    
    return df

def map_binary_labels(df):
    """Maps multiclass labels to binary (0 = Normal, 1 = Attack)."""
    print("Mapping labels to binary...")
    if 'label' not in df.columns:
        raise ValueError("Could not find 'label' column. Columns present: " + str(df.columns))
        
    # Check what the actual 'Normal' string is. usually 'BENIGN'
    benign_label = 'BENIGN'
    
    df['is_attack'] = (df['label'] != benign_label).astype(int)
    
    # Log the class distribution
    counts = df['is_attack'].value_counts()
    print(f"Binary Class Distribution:\n0 (Normal): {counts.get(0, 0)}\n1 (Attack): {counts.get(1, 0)}")
    return df


def split_and_scale(df):
    """Performs stratified train/test split and robust scaling to avoid data leakage."""
    print("Splitting and scaling data (80/20 train/test split)...")
    
    # Separate features and target
    X = df.drop(columns=['label', 'is_attack'])
    y = df['is_attack']
    
    # Retain the original string label for multiclass evaluation later if needed,
    # but for purely binary MVP we exclude it from the feature set.
    original_labels = df['label']
    
    # Drop non-predictive string columns and ports to prevent leakage
    cols_to_drop = ['flow_id', 'source_ip', 'destination_ip', 'timestamp', 'source_port', 'destination_port']
    # Not all files might have these exact names after cleaning, so we use `errors='ignore'`
    X = X.drop(columns=cols_to_drop, errors='ignore')


    # Ensure all features are numeric type before scaling
    for col in X.columns:
         X[col] = pd.to_numeric(X[col], errors='coerce')
         
    # Check for excessive NaNs introduced
    nan_counts = X.isna().sum()
    if nan_counts.sum() > 0:
        print(f"Warning: Coercion introduced NaNs: \n{nan_counts[nan_counts > 0]}")

         
    # Safety drop just in case coercion caused NaNs
    valid_idx = X.notna().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]
    original_labels = original_labels[valid_idx]
    
    # 80/20 Stratified Split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # We also keep the original labels attached for multiclass logic later
    labels_train = original_labels.iloc[train_idx]
    labels_test = original_labels.iloc[test_idx]
    
    # Fit the scaler ONLY on training data to prevent leakage
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Reconstruct train/test dataframes
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
