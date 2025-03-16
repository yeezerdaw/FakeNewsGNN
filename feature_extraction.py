import torch
from transformers import BertTokenizer, BertModel
import numpy as np

class BertEncoder(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert.to(self.device)

    def forward(self, text_list, batch_size=128):
        all_embeddings = []
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.bert(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)
            print(f"Processed {i+batch_size}/{len(text_list)} statements")
        return torch.tensor(np.vstack(all_embeddings), dtype=torch.float)
