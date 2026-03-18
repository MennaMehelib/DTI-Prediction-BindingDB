import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    MACCS_BITS, AAC_FEATURES, DC_FEATURES,
    STANDARD_AAS, ALL_DIPEPTIDES, OUTPUT_DIR
)


# DRUG FEATURES: MACCS Keys

def smiles_to_maccs(smiles_str):
   
    try:
        mol = Chem.MolFromSmiles(str(smiles_str))
        if mol is None:
            return None
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.array(fp)
        return arr[1:]  # Remove leading 0
    except Exception:
        return None


def compute_maccs_fingerprints(df, smiles_column='Drug'):
    
    print("\n── Drug Fingerprints: MACCS Keys ──")
    
    maccs_list, valid_indices = [], []
    for idx, smi in enumerate(tqdm(df[smiles_column], desc='MACCS Keys', ncols=80)):
        fp = smiles_to_maccs(smi)
        if fp is not None and len(fp) == MACCS_BITS:
            maccs_list.append(fp)
            valid_indices.append(idx)
    
    # Filter DataFrame
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    maccs_arr = np.array(maccs_list, dtype=np.float32)
    
    print(f"  MACCS array shape: {maccs_arr.shape}")
    print(f"  Valid rows kept  : {len(df_valid):,} ({len(valid_indices)/len(df)*100:.1f}%)")
    
    return maccs_arr, valid_indices, df_valid


# TARGET FEATURES: Amino Acid Composition (AAC)

def amino_acid_composition(sequence):
    
    seq = str(sequence).upper()
    length = len(seq)
    if length == 0:
        return np.zeros(AAC_FEATURES, dtype=np.float32)
    
    c = Counter(seq)
    return np.array([c.get(aa, 0) / length for aa in STANDARD_AAS], dtype=np.float32)


def compute_aac_features(df, sequence_column='Target'):
    
    print("\n── Target Representation: AAC ──")
    
    aac_arr = np.array(
        [amino_acid_composition(seq) for seq in 
         tqdm(df[sequence_column], desc='AAC', ncols=80)],
        dtype=np.float32
    )
    
    print(f"  AAC array shape: {aac_arr.shape}")
    return aac_arr


# TARGET FEATURES: Dipeptide Composition (DC)

def dipeptide_composition(sequence):
    
    seq = str(sequence).upper()
    n = len(seq) - 1
    if n <= 0:
        return np.zeros(DC_FEATURES, dtype=np.float32)
    
    pairs = [seq[i:i+2] for i in range(n)]
    c = Counter(pairs)
    return np.array([c.get(dp, 0) / n for dp in ALL_DIPEPTIDES], dtype=np.float32)


def compute_dc_features(df, sequence_column='Target'):
    
    print("\n── Target Representation: DC ──")
    
    dc_arr = np.array(
        [dipeptide_composition(seq) for seq in 
         tqdm(df[sequence_column], desc='DC', ncols=80)],
        dtype=np.float32
    )
    
    print(f"  DC array shape: {dc_arr.shape}")
    return dc_arr


# FEATURE COMBINATION & STANDARDIZATION

def combine_features(maccs_arr, aac_arr=None, dc_arr=None):
    
    print("\n── Feature Combinations ──")
    
    feature_sets = {}
    
    if aac_arr is not None:
        feature_sets['ACC'] = np.hstack([maccs_arr, aac_arr])
        print(f"  ACC (MACCS+AAC)     : {feature_sets['ACC'].shape}")
    
    if dc_arr is not None:
        feature_sets['DC'] = np.hstack([maccs_arr, dc_arr])
        print(f"  DC  (MACCS+DC)      : {feature_sets['DC'].shape}")
    
    if aac_arr is not None and dc_arr is not None:
        feature_sets['FULL'] = np.hstack([maccs_arr, aac_arr, dc_arr])
        print(f"  FULL (MACCS+AAC+DC) : {feature_sets['FULL'].shape}")
    
    return feature_sets


def standardize_features(X, scaler=None):
    
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)
    else:
        X_scaled = scaler.transform(X).astype(np.float32)
    
    return X_scaled, scaler


def plot_correlation_heatmap(X_scaled, n_features=30, title_suffix=""):
    
    print(f"\n── Feature Correlation Heatmap (first {n_features} features) ──")
    
    sample_idx = np.random.choice(len(X_scaled), min(1000, len(X_scaled)), replace=False)
    corr_matrix = pd.DataFrame(X_scaled[sample_idx, :n_features]).corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, cmap='coolwarm', center=0, square=True,
        linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        vmin=-1, vmax=1
    )
    ax.set_title(f'Correlation Heatmap — First {n_features} Features\n{title_suffix}',
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Kd_feature_correlation_heatmap.png", dpi=150)
    plt.close()
