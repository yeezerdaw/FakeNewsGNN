import torch
from transformers import BertTokenizer, BertModel

class BertEncoder(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert.to(self.device)

    def forward(self, texts, batch_size=128):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

            with torch.no_grad():  # ✅ No gradients needed (Faster)
                outputs = self.bert(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()  # ✅ Use CLS token embedding

            all_embeddings.append(batch_embeddings)
            print(f"Processed {i+batch_size}/{len(texts)} statements", end="\r")  # ✅ Overwrites progress

        return torch.cat(all_embeddings, dim=0)

# Usage in main.py
bert_encoder = BertEncoder()
x = bert_encoder(df["statement"].tolist())  # Extracts embeddings
