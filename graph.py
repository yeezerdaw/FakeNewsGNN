import torch

def build_graph(df):
    news_nodes = {statement: idx for idx, statement in enumerate(df["statement"])}
    edges = []
    for _, row in df.iterrows():
        speaker = row["speaker"]
        if speaker not in news_nodes:
            news_nodes[speaker] = len(news_nodes)
        edges.append((news_nodes[speaker], news_nodes[row["statement"]]))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index
