DTI Datasets (BindingDB)
        ↓
Preprocessing (Harmonize + Binarize)
        ↓
Feature Extraction
   ┌──────────────┬──────────────────────┐
   │ Drug: MACCS  │ Protein: AAC + DC     │
   └──────────────┴──────────────────────┘
        ↓
Feature Matrix X + StandardScaler
        ↓
Data Balancing (GAN)
        ↓
80/20 Split
        ↓
Models (RFC هو الأفضل)
        ↓
Performance Analysis + Metrics + Friedman Test
