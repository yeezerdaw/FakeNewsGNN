import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, GATConv

class FakeNewsGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_gat=False):
        super(FakeNewsGNN, self).__init__()
        if use_gat:
            self.conv1 = GATConv(input_dim, hidden_dim)
            self.conv2 = GATConv(hidden_dim, hidden_dim)
        else:
            self.conv1 = GraphSAGE(input_dim, hidden_dim, num_layers=3)
            self.conv2 = GraphSAGE(hidden_dim, hidden_dim, num_layers=3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x
