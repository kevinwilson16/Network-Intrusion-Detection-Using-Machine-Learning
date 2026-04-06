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

    # We NO LONGER do One-Hot Encoding or global fillna() here before the split to prevent structural leakage.
    return df


def split_and_save(df):
    print("Splitting and Scaling UNSW-NB15...")
    
    X = df.drop(columns=['attack_cat', 'Label'])
    y_multiclass = df['attack_cat']
    y_binary = df['Label']
    
    print("Replacing Infs with NaNs before the split...")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # STRATIFIED SPLIT
    X_train, X_test, y_train_multi, y_test_multi = train_test_split(
        X, y_multiclass, test_size=0.2, random_state=42, stratify=y_multiclass
    )
    
    # Convert to DataFrames directly from splits to avoid index misalignment warnings
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    
    # CATEGORICAL ENCODING
    from sklearn.preprocessing import OneHotEncoder
    categorical_cols = ['proto', 'state', 'service']
    categorical_cols = [c for c in categorical_cols if c in X_train.columns]
    
    if categorical_cols:
        print("Applying OneHotEncoding fitted strictly on X_train...")
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
        
        train_cat = ohe.fit_transform(X_train[categorical_cols])
        test_cat = ohe.transform(X_test[categorical_cols])
        
        cat_feature_names = ohe.get_feature_names_out(categorical_cols)
        train_cat_df = pd.DataFrame(train_cat, columns=cat_feature_names, index=X_train.index)
        test_cat_df = pd.DataFrame(test_cat, columns=cat_feature_names, index=X_test.index)
        
        X_train = pd.concat([X_train.drop(columns=categorical_cols), train_cat_df], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_cols), test_cat_df], axis=1)
    
    # IMPUTATION
    print("Imputing NaNs purely based on X_train statistics...")
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if X_train[col].isna().any() or X_test[col].isna().any():
            median_train_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_train_val)
            X_test[col] = X_test[col].fillna(median_train_val)
    
    # ROBUST SCALING
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Attach labels back for Parquet storage
    train_final = X_train_df.copy()
    train_final['multiclass_label'] = y_train_multi.values
    
    # Make sure we use proper alignment for numpy to dict extraction
    # Since y_binary is a Series, use .loc for absolute indexing matching train_test_split
    train_final['label'] = y_binary.loc[y_train_multi.index].values
    
    test_final = X_test_df.copy()
    test_final['multiclass_label'] = y_test_multi.values
    test_final['label'] = y_binary.loc[y_test_multi.index].values
    
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    train_final.to_parquet(os.path.join(PROCESSED_DATA_PATH, "train_unsw.parquet"))
    test_final.to_parquet(os.path.join(PROCESSED_DATA_PATH, "test_unsw.parquet"))
    
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    df = load_and_merge_unsw()
    df = clean_unsw(df)
    split_and_save(df)
