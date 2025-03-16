import torch
import pandas as pd
from datasets import load_dataset

def load_liar_dataset():
    dataset = load_dataset("liar", trust_remote_code=True)
    df = pd.DataFrame(dataset["train"])  # Use training split

    # ✅ Encode Labels
    label_mapping = {
        0: 0,  # False
        5: 1,  # Pants-fire (Extremely False)
        4: 2,  # Barely-true
        1: 3,  # Half-true
        2: 4,  # Mostly-true
        3: 5   # True
    }
    df["label"] = df["label"].map(label_mapping)
    y = torch.tensor(df["label"].values, dtype=torch.long)

    return df, y
