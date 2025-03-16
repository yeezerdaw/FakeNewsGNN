import os
import torch
import random
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data

# 🔹 Set Random Seeds for Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

# 🔹 Optimize CUDA Performance (Balances Speed & Stability)
torch.backends.cudnn.benchmark = True  # ✅ Speeds up convolutions
torch.backends.cudnn.deterministic = False  # ❌ But allows some variation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Ensures CUDA consistency

def train_model(model, x, edge_index, y, epochs=1000, lr=0.01):
    """
    Trains the GNN model using BERT embeddings as input.

    Args:
        model: The Fake News Detection GNN model.
        x: Node feature matrix (BERT embeddings).
        edge_index: Graph edges (news-speech relationships).
        y: Ground truth labels.
        epochs: Number of training iterations.
        lr: Learning rate.

    Returns:
        Trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Data(x=x, edge_index=edge_index, y=y).to(device)

    # Move model to GPU if available
    model = model.to(device)
    
    # 🔹 Optimizer & Learning Rate Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.5)  # 🔹 Reduces LR every 200 epochs
    criterion = torch.nn.CrossEntropyLoss()

    # 🔥 Training Loop
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

    # 🔹 Evaluate Model on Training Data
    model.eval()
    with torch.no_grad():
        train_predictions = model(data.x, data.edge_index).argmax(dim=1)
    
    train_acc = (train_predictions == data.y).sum().item() / y.size(0)
    print(f"Train Accuracy: {train_acc:.4f}")

    return model
