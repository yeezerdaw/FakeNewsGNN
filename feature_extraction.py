import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class BertEncoder(nn.Module):
    def __init__(self, hidden_dim=768):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(hidden_dim, hidden_dim)  # Learnable transformation layer

    def forward(self, text_list):
        inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = self.bert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Use CLS token embedding
        return self.fc(cls_embedding)  # Apply trainable linear layer


bert_encoder = BertEncoder().to(device)
