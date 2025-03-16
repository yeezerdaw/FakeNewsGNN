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

    def forward(self, text_list, batch_size=32):
    all_embeddings = []
    
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i+batch_size]

        # Tokenize batch with limited max length
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        
        with torch.no_grad():  # Prevent extra memory usage
            outputs = self.bert(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embedding.cpu())  # Move to CPU to free GPU memory

        torch.cuda.empty_cache()  # Clear unused memory

        print(f"Processed {i+batch_size}/{len(text_list)} statements")

    return torch.cat(all_embeddings, dim=0).to(device)  # Move final embeddings back to GPU



bert_encoder = BertEncoder().to(device)
