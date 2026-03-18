import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tdc.multi_pred import DTI
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

from config import (
    SEED, OUTPUT_DIR, DATASET_NAME, 
    BINARIZATION_THRESHOLDS, PRIMARY_THRESHOLD,
    PAPER_REFERENCES
)


# DATA LOADING
def load_bindingdb_kd():
    
    print("\n" + "=" * 65)
    print("DATA LOADING: BindingDB_Kd via TDC")
    print("=" * 65)
    
    data_obj = DTI(name=DATASET_NAME)
    raw_df = data_obj.get_data()
    
    print(f"Raw DataFrame shape : {raw_df.shape}")
    print(f"Columns             : {list(raw_df.columns)}")
    print(f"\nFirst 3 rows:")
    print(raw_df.head(3).to_string())
    
    return data_obj, raw_df


def perform_eda(raw_df):

    print("\n" + "=" * 65)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 65)
    
    # Basic statistics
    print("\n Basic Statistics")
    stats = {
        'total_pairs': len(raw_df),
        'unique_drugs': raw_df['Drug'].nunique(),
        'unique_proteins': raw_df['Target'].nunique()
    }
    print(f"  Total DTI pairs  : {stats['total_pairs']:>10,}")
    print(f"  Unique drugs     : {stats['unique_drugs']:>10,}")
    print(f"  Unique proteins  : {stats['unique_proteins']:>10,}")
    
    # Missing values
    print("\n Missing Values")
    print(raw_df.isnull().sum().to_string())
    
    # Kd distribution
    print("\n Kd Value Statistics ")
    print(raw_df['Y'].describe().round(2).to_string())
    
    # Plot Kd distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(raw_df['Y'].clip(0, 500), bins=100,
                 color='steelblue', edgecolor='white', alpha=0.85)
    for Th_line, col in [(10,'red'), (20,'orange'), (30,'green')]:
        axes[0].axvline(Th_line, color=col, linestyle='--', linewidth=1.8,
                        label=f'Th = {Th_line}')
    axes[0].set_title('Distribution of Kd Values (clipped at 500 nM)')
    axes[0].set_xlabel('Kd (nM)')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    
    axes[1].boxplot(raw_df['Y'].clip(0, 200), patch_artist=True,
                    boxprops=dict(facecolor='#6baed6', alpha=0.8),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].set_title('Boxplot of Kd Values (clipped at 200 nM)')
    axes[1].set_ylabel('Kd (nM)')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Kd_value_distribution.png", dpi=150)
    plt.close()
    print("\n Saved: Kd_value_distribution.png")
    
    # Sample data
    print("\n Sample Drug & Target ")
    print(f"  SMILES  : {str(raw_df['Drug'].iloc[0])[:90]}...")
    print(f"  Protein : {str(raw_df['Target'].iloc[0])[:90]}...")
    
    return stats


def binarize_dataset(thresholds=BINARIZATION_THRESHOLDS):
    
    print("\n" + "=" * 65)
    print("DATA PREPROCESSING: Binarization")
    print("=" * 65)
    
    print("\n Affinity Harmonization (mode='mean')")
    data_obj = DTI(name=DATASET_NAME)
    data_obj.harmonize_affinities(mode='mean')
    
    print("\n Binarization with thresholds ")
    binarized_dfs = {}
    
    for Th in thresholds:
        d_tmp = DTI(name=DATASET_NAME)
        d_tmp.harmonize_affinities(mode='mean')
        d_tmp.binarize(threshold=Th, order='descending')
        df_tmp = d_tmp.get_data()
        
        neg = int((df_tmp['Y'] == 0).sum())
        pos = int((df_tmp['Y'] == 1).sum())
        binarized_dfs[Th] = df_tmp
        
        paper_stats = PAPER_REFERENCES['dataset_stats'][Th]
        print(f"  Th={Th:2d}  Class 0: {neg:6,}  Class 1: {pos:5,}  "
              f"| Paper: Neg={paper_stats['neg']:,}  Pos={paper_stats['pos']:,}")
    
    return binarized_dfs


def plot_class_distribution(df_main, title_prefix="BEFORE GAN"):
    
    neg = int((df_main['Y'] == 0).sum())
    pos = int((df_main['Y'] == 1).sum())
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Bar plot
    axes[0].bar(['Class 0 (No Interaction)', 'Class 1 (Interaction)'],
                [neg, pos], color=['#d62728', '#1f77b4'], 
                edgecolor='white', width=0.45)
    axes[0].set_title(f'Class Distribution {title_prefix}\n(BindingDB-Kd, Th={PRIMARY_THRESHOLD})')
    axes[0].set_ylabel('Sample Count')
    axes[0].set_ylim(0, neg * 1.15)
    for i, v in enumerate([neg, pos]):
        axes[0].text(i, v + 200, f'{v:,}', ha='center', fontsize=10, fontweight='bold')
    
    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        [neg, pos],
        labels=[f'Class 0\n{neg:,}', f'Class 1\n{pos:,}'],
        colors=['#d62728', '#1f77b4'],
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 11},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    axes[1].set_title(f'Class Proportion {title_prefix}\n(BindingDB-Kd, Th={PRIMARY_THRESHOLD})')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Kd_class_distribution_{title_prefix.replace(' ', '_').lower()}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()


def train_test_split_stratified(X, y, test_size=0.2):
    
    return train_test_split(
        X, y, test_size=test_size, 
        random_state=SEED, stratify=y
    )