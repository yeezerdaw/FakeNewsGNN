import torch
from dataset import load_liar_dataset
from feature_extraction import BertEncoder
from graph import build_graph
from model import FakeNewsGNN
from train import train_model

# ✅ Load Dataset
df, y = load_liar_dataset()  # Ensure df is defined

# ✅ Extract BERT Features
bert_encoder = BertEncoder()  # Initialize BERT Model
x = bert_encoder(df["statement"].tolist())  # Extract Features

# ✅ Build Graph
edge_index = build_graph(df)

# ✅ Initialize & Train Model
model = FakeNewsGNN(input_dim=768, hidden_dim=128, output_dim=6)
model = train_model(model, x, edge_index, y)

# ✅ Evaluate Model
model.eval()
with torch.no_grad():
    predictions = model(x.to("cuda" if torch.cuda.is_available() else "cpu"), edge_index).argmax(dim=1)

accuracy = (predictions == y.to(predictions.device)).sum().item() / y.size(0)
print(f"Test Accuracy: {accuracy:.4f}")
