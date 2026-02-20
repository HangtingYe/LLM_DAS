import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(input_dim, hidden_dim),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_dim, 1),
            # nn.Sigmoid()

            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class Wrapper:
    def __init__(self, input_dim, hidden_dim=256, lr=1e-3, n_epochs=100, device=None):
        # 自动检测 GPU
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = MLPBinaryClassifier(input_dim, hidden_dim).to(self.device)
        self.lr = lr
        self.n_epochs = n_epochs
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X_train, y_train):
        X = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32, device=self.device)
        for _ in range(self.n_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

    def predict_score(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs = self.model(X).cpu().numpy().ravel()
        return probs

    def predict(self, X, threshold=0.5):
        return (self.predict_score(X) >= threshold).astype(int)
