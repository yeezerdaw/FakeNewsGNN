import torch
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
import random
import numpy as np
import os

# ✅ Fix CUDA Non-Determinism
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Ensures CUDA reproducibility

def train_model(model, x, edge_index, y, epochs=1000, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Data(x=x, edge_index=edge_index, y=y).to(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # L2 Regularization
    scheduler = StepLR(optimizer, step_size=200, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # ✅ Evaluate Model on Training Data AFTER Training
    model.eval()
    with torch.no_grad():
        train_predictions = model(data.x, data.edge_index).argmax(dim=1)

    train_acc = (train_predictions == data.y).sum().item() / data.y.size(0)
    print(f"✅ Final Train Accuracy: {train_acc:.4f}")

    return model
