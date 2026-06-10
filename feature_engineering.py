import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from collections import Counter
from tqdm import tqdm   

STANDARD_AAS = list('ACDEFGHIKLMNPQRSTVWY')

def smiles_to_maccs(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp)[1:]
    except:
        return None

def amino_acid_composition(seq):
    seq = str(seq).upper()
    length = len(seq)
    if length == 0:
        return np.zeros(20)

    c = Counter(seq)
    return np.array([c.get(aa, 0)/length for aa in STANDARD_AAS])

def build_features(df, max_samples=5000):  
    X, y = [], []

    df = df.head(max_samples)  

    for smi, seq, label in tqdm(
        zip(df['Drug'], df['Target'], df['Y']),
        total=len(df),
        desc="Building features"
    ):
        maccs = smiles_to_maccs(smi)
        if maccs is None:
            continue

        aac = amino_acid_composition(seq)

        features = np.concatenate([maccs, aac])
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)