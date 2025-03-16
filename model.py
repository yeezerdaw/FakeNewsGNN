import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE

class FakeNewsGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GraphSAGE(input_dim, hidden_dim)
        self.conv2 = GraphSAGE(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
