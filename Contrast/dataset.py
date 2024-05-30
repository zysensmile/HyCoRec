import dgl
import dgl.data
import math
import torch
import numpy as np
import pickle as pkl
from torch_geometric.data import Data


def batcher(batch):
    graph_q, graph_k = zip(*batch)
    return list(graph_q), list(graph_k)


class GraphDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            graph_path=None,
            meta_path=None,
            random_walk_hops=64,
            restart_prob=0.8,
            sample_num=10240
    ):
        super(GraphDataset).__init__()
        graph_list, _ = dgl.data.utils.load_graphs(graph_path)
        self.graph = graph_list[0]
        self.node_num = self.graph.number_of_nodes()
        self.random_walk_hops = random_walk_hops
        self.restart_prob = restart_prob
        self.sample_num = sample_num
        self.total = sample_num
        with open(meta_path, 'rb') as f:
            meta_data = pkl.load(f)
            self.final_entity2id = meta_data['final_entity2id']
            self.final_id2entity = meta_data['final_id2entity']
            self.up2final = meta_data['up2final']
            self.n_embedding = len(self.final_entity2id)
        return

    def __len__(self):
        return self.sample_num

    def __iter__(self):
        degrees = self.graph.in_degrees().double() ** 0.75
        prob = degrees / torch.sum(degrees)
        samples = np.random.choice(self.node_num, self.sample_num, replace=True, p=prob.numpy())
        for idx in samples:
            yield self.__getitem__(idx)

    def __getitem__(self, node_index):
        max_nodes_per_seed = max(
            self.random_walk_hops,
            int(
                (
                        (self.graph.in_degrees(node_index) ** 0.75)
                        * math.e
                        / (math.e - 1)
                        / self.restart_prob
                )
                + 0.5
            ),
        )

        max_nodes_per_seed = min(max_nodes_per_seed, 200)  # avoid too large subgraph

        traces = dgl.contrib.sampling.random_walk_with_restart(
            self.graph,
            seeds=[node_index, node_index],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=max_nodes_per_seed,
        )

        graph_q = self.trace_to_pyg_graph(
            seed=node_index,
            trace=traces[0]
        )
        graph_k = self.trace_to_pyg_graph(
            seed=node_index,
            trace=traces[1]
        )

        return graph_q, graph_k

    def trace_to_pyg_graph(self, seed, trace):
        vertices = set(torch.cat(trace).tolist())
        vertices.add(seed)
        sub_graph = self.graph.subgraph(list(vertices))

        # mapping up_id to final_id
        node_up_id = sub_graph.parent_nid.tolist()
        node_final_id = list()
        for up_id in node_up_id:
            node_final_id.append(self.up2final[up_id])
        node_final_id = torch.tensor(node_final_id, dtype=torch.int64)

        # get edge_type slices
        edge_type = self.graph.edata['edge_type'][sub_graph.parent_eid]

        # create pyg graph
        edge_index = torch.stack(sub_graph.edges(), dim=0)
        pyg_graph = Data(edge_index=edge_index, node_final_id=node_final_id, edge_type=edge_type, seed=seed)

        return pyg_graph
