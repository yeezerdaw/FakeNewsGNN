import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# Set device and enable optimizations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True  # Improve performance
torch.cuda.empty_cache()  # Clear memory

# Load BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class BertEncoder(nn.Module):
    def __init__(self, hidden_dim=768):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased").to(device).half()  # Convert to FP16
        self.fc = nn.Linear(hidden_dim, hidden_dim).to(device)

    def forward(self, text_list, batch_size=8):  # Reduce batch size further
        all_embeddings = []
        
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i+batch_size]

            # Tokenize with max_length=64 to reduce memory usage
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)

            with torch.no_grad(), torch.cuda.amp.autocast():  # Enable mixed precision
                outputs = self.bert(**inputs)

            cls_embedding = outputs.last_hidden_state[:, 0, :].half()  # Convert to FP16
            all_embeddings.append(cls_embedding.cpu())  # Move to CPU to free GPU memory

            torch.cuda.empty_cache()  # Force clear CUDA memory

            print(f"Processed {i+batch_size}/{len(text_list)} statements")

        return torch.cat(all_embeddings, dim=0).to(device)  # Move final embeddings back to GPU

# Instantiate encoder
bert_encoder = BertEncoder()
