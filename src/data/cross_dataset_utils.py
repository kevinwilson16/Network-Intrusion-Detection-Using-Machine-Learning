import os
import glob
import pandas as pd
import numpy as np

# Unified Feature Names
SHARED_FEATURES = [
    'duration',             # Microseconds
    'src_packets',          # Count
    'dst_packets',          # Count
    'src_bytes',            # Bytes
    'dst_bytes',            # Bytes
    'src_mean_packet_sz',   # Bytes
    'dst_mean_packet_sz',   # Bytes
    'src_iat_mean',         # Microseconds
    'dst_iat_mean'          # Microseconds
]

def load_and_map_cic():
    print("Loading and mapping CIC-IDS2017 raw data...")
    raw_path = "data/cicids2017/raw"
    all_files = glob.glob(os.path.join(raw_path, "*.csv"))
    
    mapping = {
        'flow duration': 'duration',
        'total fwd packets': 'src_packets',
        'total backward packets': 'dst_packets',
        'total length of fwd packets': 'src_bytes',
        'total length of bwd packets': 'dst_bytes',
        'fwd packet length mean': 'src_mean_packet_sz',
        'bwd packet length mean': 'dst_mean_packet_sz',
        'fwd iat mean': 'src_iat_mean',
        'bwd iat mean': 'dst_iat_mean',
        'label': 'label'
    }
    
    dfs = []
    for f in all_files:
        print(f"Reading {os.path.basename(f)}...")
        df = pd.read_csv(f, skipinitialspace=True, encoding='latin1', low_memory=False)
        # Lowercase and strip columns for consistent mapping
        df.columns = df.columns.str.strip().str.lower()
        
        # Select and rename columns
        existing_cols = [c for c in mapping.keys() if c in df.columns]
        df_sub = df[existing_cols].copy()
        df_sub.rename(columns=mapping, inplace=True)
        dfs.append(df_sub)
        
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Binary Label Mapping
    final_df['is_attack'] = (final_df['label'] != 'BENIGN').astype(int)
    final_df.drop(columns=['label'], inplace=True)
    
    # Cleaning
    print("Coercing CIC-IDS2017 shared features to numeric...")
    numeric_cols = [c for c in SHARED_FEATURES if c in final_df.columns]
    for col in numeric_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        # We DO NOT impute or handle infinity here to prevent data leakage.
    
    # Units: CIC-IDS2017 duration and IAT are already in Microseconds.
    
    os.makedirs("data/cross_dataset", exist_ok=True)
    final_df.to_parquet("data/cross_dataset/cic_shared_raw.parquet")
    print(f"CIC Shared Raw generated: {len(final_df)} rows.")

def load_and_map_unsw():
    print("Loading and mapping UNSW-NB15 raw data...")
    raw_path = "data/unsw-nb15/raw"
    
    unsw_cols_all = [
        'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 
        'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 
        'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 
        'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 
        'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 
        'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 
        'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
        'ct_dst_src_ltm', 'attack_cat', 'Label'
    ]
    
    mapping = {
        'dur': 'duration',
        'Spkts': 'src_packets',
        'Dpkts': 'dst_packets',
        'sbytes': 'src_bytes',
        'dbytes': 'dst_bytes',
        'smeansz': 'src_mean_packet_sz',
        'dmeansz': 'dst_mean_packet_sz',
        'Sintpkt': 'src_iat_mean',
        'Dintpkt': 'dst_iat_mean',
        'Label': 'is_attack'
    }
    
    dfs = []
    for i in range(1, 5):
        file_path = os.path.join(raw_path, f"UNSW-NB15_{i}.csv")
        print(f"Reading {os.path.basename(file_path)}...")
        df = pd.read_csv(file_path, header=None, names=unsw_cols_all, low_memory=False, encoding='latin-1')
        
        # Select and rename
        df_sub = df[list(mapping.keys())].copy()
        df_sub.rename(columns=mapping, inplace=True)
        dfs.append(df_sub)
        
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Cleaning
    print("Cleaning UNSW-NB15 shared features...")
    final_df.replace([' ', '-'], np.nan, inplace=True)
    
    numeric_cols = [c for c in SHARED_FEATURES if c in final_df.columns]
    for col in numeric_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        # We DO NOT impute or handle infinity here to prevent data leakage.
    
    # UNIT CONVERSION TO MATCH CIC-IDS2017 (Microseconds)
    print("Applying unit conversions (Seconds and Milliseconds -> Microseconds)...")
    final_df['duration'] = final_df['duration'] * 1e6
    final_df['src_iat_mean'] = final_df['src_iat_mean'] * 1e3
    final_df['dst_iat_mean'] = final_df['dst_iat_mean'] * 1e3
    
    os.makedirs("data/cross_dataset", exist_ok=True)
    final_df.to_parquet("data/cross_dataset/unsw_shared_raw.parquet")
    print(f"UNSW Shared Raw generated: {len(final_df)} rows.")

if __name__ == "__main__":
    load_and_map_cic()
    load_and_map_unsw()
