import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, noise_dim, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, z):
        return self.net(z)


def balance_data(X, y, noise_dim=100, epochs=200):
    X_min = X[y == 1]
    X_maj = X[y == 0]

    n_to_generate = len(X_maj) - len(X_min)

    if n_to_generate <= 0:
        return X, y

    G = Generator(noise_dim, X.shape[1])
    optimizer = optim.Adam(G.parameters(), lr=0.001)

    for _ in range(epochs):
        z = torch.randn(len(X_min), noise_dim)
        fake = G(z)

        loss = ((fake - torch.tensor(X_min, dtype=torch.float32))**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    z = torch.randn(n_to_generate, noise_dim)
    synthetic = G(z).detach().numpy()

    X_bal = np.vstack([X, synthetic])
    y_bal = np.concatenate([y, np.ones(n_to_generate)])

    return X_bal, y_bal