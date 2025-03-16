import torch
from dataset import load_liar_dataset
from feature_extraction import bert_encoder
from graph import build_graph
from model import FakeNewsGNN
from train import train_model

# Detect Device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Dataset
df, y = load_liar_dataset()


x = bert_encoder(df["statement"].tolist(), batch_size=8)



# Build Graph
edge_index = build_graph(df)

# Move Tensors to Device
x = torch.tensor(x, dtype=torch.float).to(device)  
edge_index = edge_index.to(device)
y = y.to(device)

# Initialize & Train Model on Device
model = FakeNewsGNN(input_dim=768, hidden_dim=128, output_dim=6).to(device)
model = train_model(model, x, edge_index, y)

# Evaluate Model
model.eval()
with torch.no_grad():
    predictions = model(x, edge_index).argmax(dim=1)

# Compute Accuracy
accuracy = (predictions == y).sum().item() / y.size(0)
print(f"Test Accuracy: {accuracy:.4f}")
