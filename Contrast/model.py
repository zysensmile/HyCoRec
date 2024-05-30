import torch
from torch import nn
from tqdm import tqdm
from dataset import GraphDataset, batcher
from torch.nn.functional import normalize
from torch_geometric.nn import RGCNConv


class GraphEncoder(nn.Module):
    def __init__(self, n_embedding, emb_dim=128, n_relation=39, n_bases=8):
        super(GraphEncoder, self).__init__()
        self.n_embedding = n_embedding
        self.emb_dim = emb_dim
        self.n_relation = n_relation
        self.n_bases = n_bases
        self.encoder = RGCNConv(self.emb_dim, self.emb_dim, self.n_relation, self.n_bases)
        self.embedding = nn.Embedding(self.n_embedding, self.emb_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=self.emb_dim ** -0.5)

    def forward(self, batch_graph):
        batch_feature = list()
        for graph in batch_graph:
            node_embedding = torch.index_select(self.embedding.weight, 0, graph.node_final_id)
            node_feature = self.encoder(node_embedding, graph.edge_index, graph.edge_type)
            graph_embedding = torch.sum(node_feature, dim=0)
            graph_feature = normalize(graph_embedding, p=2, dim=-1, eps=1e-5)
            batch_feature.append(graph_feature)
        return torch.stack(batch_feature, dim=0)


if __name__ == '__main__':
    dataset = GraphDataset(
        random_walk_hops=128,
        restart_prob=0.5,
        sample_num=1024
    )
    print(f"load {dataset.__class__.__name__}")

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=32,
        collate_fn=batcher
    )
    print(f"load {loader.__class__.__name__}")

    encoder = GraphEncoder(n_embedding=dataset.n_embedding)
    for batch_q, batch_k in tqdm(loader):
        feat_q = encoder(batch_q)
        feat_k = encoder(batch_k)
