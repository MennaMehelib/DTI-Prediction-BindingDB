import os
import random
import numpy as np
import torch

# GLOBAL CONFIGURATION
SEED = 42
OUTPUT_DIR = "outputs"

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Dataset configuration
DATASET_NAME = 'BindingDB_Kd'
BINARIZATION_THRESHOLDS = [10, 20, 30]
PRIMARY_THRESHOLD = 10

# Feature dimensions
MACCS_BITS = 166
AAC_FEATURES = 20
DC_FEATURES = 400

# Feature combinations
FEATURE_DIMS = {
    'ACC': MACCS_BITS + AAC_FEATURES,      # 186
    'DC': MACCS_BITS + DC_FEATURES,        # 566
    'FULL': MACCS_BITS + AAC_FEATURES + DC_FEATURES  # 586
}

# GAN configuration
GAN_CONFIG = {
    'noise_dim': 100,
    'epochs': 500,
    'batch_size': 64,
    'learning_rate': 0.0002,
    'generator_layers': [256, 512],
    'discriminator_layers': [512, 256]
}

# Model configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': SEED,
    'dl_epochs': 50,
    'dl_batch_size': 256,
    'dl_learning_rate': 0.001,
    'fcnn_layers': [128, 64, 32],
    'dropout_rate': 0.3,
    'mha_embed_dim': 128,
    'mha_num_heads': 4
}

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Standard amino acids for composition features
STANDARD_AAS = list('ACDEFGHIKLMNPQRSTVWY')
ALL_DIPEPTIDES = [''.join(p) for p in __import__('itertools').product(STANDARD_AAS, repeat=2)]

# Paper reference values (for validation)
PAPER_REFERENCES = {
    'dataset_stats': {
        10: {'neg': 38910, 'pos': 3326},
        20: {'neg': 37915, 'pos': 4321},
        30: {'neg': 37385, 'pos': 4851}
    },
    'rfc_th10': {
        'Accuracy': 97.46, 'Precision': 97.49, 'Sensitivity': 97.46,
        'Specificity': 98.82, 'F1-Score': 97.46, 'ROC-AUC': 99.42,
        'Kappa': 94.91, 'MCC': 94.95
    }
}