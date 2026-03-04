import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Paths
RAW_DATA_PATH = "data/unsw-nb15/raw"
PROCESSED_DATA_PATH = "data/unsw-nb15/processed"

def load_and_merge_unsw():
    print("Loading UNSW-NB15 raw files...")
    
    # Define Column Names from NUSW-NB15_features.csv
    cols = [
        'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 
        'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 
        'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 
        'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 
        'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 
        'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 
        'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
        'ct_dst_src_ltm', 'attack_cat', 'Label'
    ]
    
    dfs = []
    for i in range(1, 5):
        file_path = os.path.join(RAW_DATA_PATH, f"UNSW-NB15_{i}.csv")
        print(f"Reading {file_path}...")
        # Note: Raw files do not have headers, and use latin-1 encoding usually
        df = pd.read_csv(file_path, header=None, names=cols, low_memory=False, encoding='latin-1')
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def clean_unsw(df):
    print("Cleaning data (Leakage Prevention & Robustness)...")
    
    # 1. Drop Leaky Features
    # IPs and Ports are specific to the environment
    # Timestamps (Stime, Ltime) are temporal leakage
    leaky_cols = ['srcip', 'sport', 'dstip', 'dsport', 'Stime', 'Ltime']
    df.drop(columns=leaky_cols, inplace=True, errors='ignore')
    
    # 2. Fix malformed labels and characters
    # In UNSW-NB15, attack_cat can have NaNs for 'Normal' rows
    df['attack_cat'] = df['attack_cat'].fillna('Normal').str.strip()
    
    # Handle empty strings or dashes often found in some versions of this dataset
    print("Replacing malformed characters (' ', '-') with NaN...")
    df.replace([' ', '-'], np.nan, inplace=True)
    
    # Convert obvious numeric columns to float (they might be object due to the spaces)
    # Identifying potential numeric columns (excluding categorical ones)
    potential_numeric = [
        'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'Sload', 'Dload', 
        'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 
        'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Sintpkt', 'Dintpkt', 'tcprtt', 
        'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 
        'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 
        'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm'
    ]
    for col in potential_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Identify non-numeric columns for separate handling
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 3. Handle Infinity (Cap to max finite value)
    print("Capping Infinity...")
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            max_val = df.loc[np.isfinite(df[col]), col].max()
            df[col] = df[col].replace([np.inf, -np.inf], max_val)
            
    # 4. Handle NaNs (Median Imputation)
    print("Imputing NaNs...")
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

            
    # 5. One-Hot Encoding for categorical features (proto, state, service)
    print("Encoding categorical features...")
    df = pd.get_dummies(df, columns=['proto', 'state', 'service'], drop_first=True)
    
    # CRITICAL: get_dummies can introduce NaNs if there are mismatched indices or 
    # if new columns are created. We must ensure the final feature set is clean.
    df.fillna(0, inplace=True)
    
    return df


def split_and_save(df):
    print("Splitting and Scaling...")
    
    X = df.drop(columns=['attack_cat', 'Label'])
    y_multiclass = df['attack_cat']
    y_binary = df['Label']
    
    # Stratified Split
    X_train, X_test, y_train_multi, y_test_multi = train_test_split(
        X, y_multiclass, test_size=0.2, random_state=42, stratify=y_multiclass
    )
    
    # Scale numerical features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Attach labels back for Parquet storage
    train_final = X_train_df.copy()
    train_final['multiclass_label'] = y_train_multi.values
    train_final['label'] = y_binary.iloc[y_train_multi.index].values # Map binary label back
    
    test_final = X_test_df.copy()
    test_final['multiclass_label'] = y_test_multi.values
    test_final['label'] = y_binary.iloc[y_test_multi.index].values
    
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    train_final.to_parquet(os.path.join(PROCESSED_DATA_PATH, "train_unsw.parquet"))
    test_final.to_parquet(os.path.join(PROCESSED_DATA_PATH, "test_unsw.parquet"))
    
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    df = load_and_merge_unsw()
    df = clean_unsw(df)
    split_and_save(df)
