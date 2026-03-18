import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from config import MODEL_CONFIG

# SCIKIT-LEARN MODELS

def get_sklearn_model(model_name, random_state=None):

    if random_state is None:
        random_state = MODEL_CONFIG['random_state']
    
    models = {
        'DTC': DecisionTreeClassifier(random_state=random_state),
        'RFC': RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=random_state
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=tuple(MODEL_CONFIG['fcnn_layers']),
            activation='relu', solver='adam',
            learning_rate_init=MODEL_CONFIG['dl_learning_rate'],
            max_iter=300, random_state=random_state
        )
    }
    
    return models.get(model_name)


# PYTORCH DEEP LEARNING MODELS

class FCNN(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        layers = MODEL_CONFIG['fcnn_layers']
        dropout = MODEL_CONFIG['dropout_rate']
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, layers[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(layers[0], layers[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(layers[1], layers[2]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(layers[2], 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


class MHABlock(nn.Module):
    def __init__(self, embed_dim=None, num_heads=None):
        super().__init__()
        if embed_dim is None:
            embed_dim = MODEL_CONFIG['mha_embed_dim']
        if num_heads is None:
            num_heads = MODEL_CONFIG['mha_num_heads']
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x_3d = x.unsqueeze(1)
        out, _ = self.attn(x_3d, x_3d, x_3d)
        return self.norm(x + self.dropout(out.squeeze(1)))


class MHAFCNN(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        embed_dim = MODEL_CONFIG['mha_embed_dim']
        layers = MODEL_CONFIG['fcnn_layers']
        dropout = MODEL_CONFIG['dropout_rate']
        
        self.proj = nn.Linear(input_dim, embed_dim)
        self.mha = MHABlock()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, layers[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(layers[1], layers[2]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(layers[2], 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        x = torch.relu(self.proj(x))
        x = self.mha(x)
        return self.fc(x)


# TRAINING UTILITIES
def train_torch_model(model, X_tr, y_tr, 
                      epochs=MODEL_CONFIG['dl_epochs'],
                      batch_size=MODEL_CONFIG['dl_batch_size'],
                      lr=MODEL_CONFIG['dl_learning_rate']):
    
    device = torch.device('cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    X_t = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_tr.astype(np.float32)).unsqueeze(1).to(device)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    
    return model


def predict_torch(model, X_te):
 
    device = torch.device('cpu')
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_te.astype(np.float32)).to(device)
        proba = model(X_t).cpu().numpy().flatten()
    return (proba >= 0.5).astype(int), proba