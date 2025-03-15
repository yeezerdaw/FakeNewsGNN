import torch

def build_graph(df):
    news_nodes = {statement: idx for idx, statement in enumerate(df["statement"])}
    speaker_nodes = {}
    edges = []

    for _, row in df.iterrows():
        statement_idx = news_nodes.get(row["statement"])
        if statement_idx is not None:
            if row["speaker"] not in speaker_nodes:
                speaker_nodes[row["speaker"]] = len(news_nodes) + len(speaker_nodes)
            speaker_idx = speaker_nodes[row["speaker"]]
            edges.append((speaker_idx, statement_idx))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    valid_mask = edge_index < max(news_nodes.values()) + 1
    edge_index = edge_index[:, valid_mask.all(dim=0)]

    return edge_index
