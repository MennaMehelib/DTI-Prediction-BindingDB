## Pipeline Overview

```mermaid
flowchart TD
    A[DTI Datasets<br>BindingDB] --> B[Preprocessing<br>Harmonize + Binarize]
    B --> C[Feature Extraction]
    
    subgraph Feature Extraction
        C --> D[Drug: MACCS]
        C --> E[Protein: AAC + DC]
    end
    
    C --> F[Feature Matrix X<br>+ StandardScaler]
    F --> G[Data Balancing<br>GAN]
    G --> H[80/20 Split]
    H --> I[Models<br>RFC هو الأفضل]
    I --> J[Performance Analysis<br>Metrics + Friedman Test]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style J fill:#bbf,stroke:#333,stroke-width:2px
