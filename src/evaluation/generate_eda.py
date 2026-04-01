import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def set_style():
    sns.set_theme(style="whitegrid", palette="viridis")
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def get_feature_importance(df, target_col='label', n_samples=100000, top_n=10):
    """Calculates top N feature importances using RandomForest on a numeric subset."""
    # Subsample for speed
    if len(df) > n_samples:
        df_sample = df.sample(n=n_samples, random_state=42)
    else:
        df_sample = df.copy()
        
    # Drop non-numeric for RF
    y = df_sample[target_col]
    
    # Check if target is string, encode if necessary
    if y.dtype == 'object' or pd.api.types.is_string_dtype(y):
        y = LabelEncoder().fit_transform(y)
        
    X = df_sample.select_dtypes(include=[np.number])
    # Drop the target if it's numeric and in X
    if target_col in X.columns:
        X = X.drop(columns=[target_col])
    # Also drop obvious non-features if present
    drop_cols = [c for c in X.columns if 'label' in c.lower() or 'cat' in c.lower() or 'id' in c.lower()]
    X = X.drop(columns=drop_cols, errors='ignore')
    
    # Fill any NaNs
    X = X.fillna(X.median(numeric_only=True))
    # Replace inf
    X = X.replace([np.inf, -np.inf], 1e9)
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    return importances.sort_values(ascending=False).head(top_n)

def plot_multiclass(df, label_col, dataset_name, exclude_classes, title, filename):
    plt.figure(figsize=(12, 6))
    
    # Filter out the excluded classes
    mask = ~df[label_col].isin(exclude_classes)
    filtered_df = df[mask].copy()
    
    # Check if filtered is empty
    if filtered_df.empty:
        print(f"Warning: Multiclass filtered for {dataset_name} is empty.")
        plt.close()
        return
        
    # Standardize string representations
    filtered_df['class_str'] = filtered_df[label_col].astype(str)
    
    ax = sns.countplot(
        data=filtered_df, 
        y='class_str', 
        order=filtered_df['class_str'].value_counts().index,
        hue='class_str',
        palette='magma',
        legend=False
    )
    plt.title(title, pad=20)
    plt.xlabel('Frequency')
    plt.ylabel('Attack Class')
    
    # Add labels
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(format(width, ',.0f'),
                        (width, p.get_y() + p.get_height() / 2.),
                        ha="left", va="center", xytext=(5, 0), textcoords="offset points")
            
    plt.tight_layout()
    plt.savefig(f'artifacts/plots/{filename}', dpi=300)
    plt.close()

def plot_importance(importances, title, filename):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=importances.values, 
        y=importances.index, 
        hue=importances.index,
        palette='viridis',
        legend=False
    )
    plt.title(title, pad=20)
    plt.xlabel('Gini Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'artifacts/plots/{filename}', dpi=300)
    plt.close()

def generate_section_a():
    print("--- SECTION A: Individual Dataset Analysis ---")
    
    # CIC-IDS2017
    print("Processing CIC-IDS2017 multiclass and feature importance...")
    try:
        cic = pd.read_parquet('data/cicids2017/processed/train_multiclass.parquet')
        
        # 1. Multiclass (exclude BENIGN / Normal)
        exclude_cic = ['BENIGN', 'Normal', '0', 0]
        plot_multiclass(
            cic, 'multiclass_label', 'CIC-IDS2017', exclude_cic,
            'CIC-IDS2017 Attack Distribution (Excluding BENIGN)',
            'cic_multiclass.png'
        )
        
        # 2. Feature Importance
        imp_cic = get_feature_importance(cic, target_col='label')
        plot_importance(imp_cic, 'CIC-IDS2017 Top 10 Feature Importance', 'cic_importance.png')
        del cic
    except Exception as e:
        print(f"Error processing CIC-IDS2017: {e}")

    # UNSW-NB15
    print("Processing UNSW-NB15 multiclass and feature importance...")
    try:
        unsw = pd.read_parquet('data/unsw-nb15/processed/train_unsw.parquet')
        
        # 3. Multiclass (exclude Normal / 0)
        exclude_unsw = ['Normal', '0', 0, 'BENIGN']
        plot_multiclass(
            unsw, 'multiclass_label', 'UNSW-NB15', exclude_unsw,
            'UNSW-NB15 Attack Distribution (Excluding Normal)',
            'unsw_multiclass.png'
        )
        
        # 4. Feature Importance
        imp_unsw = get_feature_importance(unsw, target_col='label')
        plot_importance(imp_unsw, 'UNSW-NB15 Top 10 Feature Importance', 'unsw_importance.png')
        del unsw
    except Exception as e:
        print(f"Error processing UNSW-NB15: {e}")

def generate_section_b():
    print("\n--- SECTION B: Cross-Dataset Analysis ---")
    shared_features = [
        'duration', 'src_packets', 'dst_packets', 'src_bytes', 'dst_bytes', 
        'src_mean_packet_sz', 'dst_mean_packet_sz', 'src_iat_mean', 'dst_iat_mean'
    ]
    
    data_paths = {
        'CIC-IDS2017': 'data/cross_dataset/cic_shared_raw.parquet',
        'UNSW-NB15': 'data/cross_dataset/unsw_shared_raw.parquet'
    }
    
    combined_df = []
    
    for name, path in data_paths.items():
        try:
            df = pd.read_parquet(path)
            df['Dataset'] = name
            combined_df.append(df)
        except Exception as e:
            print(f"Error reading {name} shared raw: {e}")
            
    if not combined_df:
        print("No shared dataset found, skipping Section B.")
        return
        
    df = pd.concat(combined_df, ignore_index=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=shared_features + ['is_attack'], inplace=True)
    
    df['Traffic Type'] = df['is_attack'].map({0: 'Normal', 1: 'Attack'})
    
    # 5. Target Variable Distribution (Class Imbalance)
    print("5. Plotting Cross-Dataset Target Variable Distribution...")
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(
        data=df, x='Dataset', hue='Traffic Type', palette='Set1'
    )
    plt.title('Cross-Dataset Target Variable Distribution (Class Imbalance)', pad=20)
    plt.ylabel('Frequency')
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(format(height, ',.0f'),
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 5), textcoords="offset points")
                        
    plt.tight_layout()
    plt.savefig('artifacts/plots/cross_imbalance.png', dpi=300)
    plt.close()
    
    # 6. Feature Distribution (Benign vs. Attack)
    print("6. Plotting Feature Distribution...")
    features_to_plot = ['duration', 'src_bytes', 'dst_bytes']
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(features_to_plot, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(
            data=df, x='Dataset', y=feature, hue='Traffic Type', palette='Set2'
        )
        plt.yscale('symlog')
        plt.title(f'{feature} Distribution\n(Normal vs Attack)')
        plt.ylabel(f'{feature} (Symlog Scale)')
        if i > 1:
            plt.legend([],[], frameon=False) # Only show legend on first
            
    plt.tight_layout()
    plt.savefig('artifacts/plots/cross_distribution.png', dpi=300)
    plt.close()
    
    # 7. Feature Correlation Heatmap
    print("7. Plotting Shared Feature Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    # Select only the shared numeric features
    corr = df[shared_features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
        vmax=1, vmin=-1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .75}
    )
    plt.title('Feature Correlation Heatmap (9 Shared Features)', pad=20)
    plt.tight_layout()
    plt.savefig('artifacts/plots/cross_correlation.png', dpi=300)
    plt.close()

def main():
    os.makedirs('artifacts/plots', exist_ok=True)
    set_style()
    generate_section_a()
    generate_section_b()
    print("All required plots generated successfully.")

if __name__ == "__main__":
    main()
