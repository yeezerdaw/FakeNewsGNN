import torch
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data

def train_model(model, x, edge_index, y, epochs=1000, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Data(x=x, edge_index=edge_index, y=y).to(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.5)  # ✅ Adjusts LR every 200 epochs
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

    # 🔹 Evaluate Model on Training Data
    model.eval()
    with torch.no_grad():
        train_predictions = model(data.x, data.edge_index).argmax(dim=1)

    train_acc = (train_predictions == data.y).sum().item() / y.size(0)
    print(f"Train Accuracy: {train_acc:.4f}")

    return model
