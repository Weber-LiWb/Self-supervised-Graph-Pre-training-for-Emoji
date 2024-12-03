
import math
import operator

import dgl
import dgl.data
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from dgl.data import AmazonCoBuy, Coauthor
import gcc.datasets.data_util as data_util
# import data_util as data_util
import random
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.graphs, _ = dgl.data.utils.load_graphs(
        dataset.dgl_graphs_file, dataset.jobs[worker_id]
    )
    dataset.length = sum([g.number_of_nodes() for g in dataset.graphs])
    # for g in dataset.graphs:
    #     g.create_formats_()
        
    np.random.seed(worker_info.seed % (2 ** 32))


class LoadBalanceGraphDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        rw_hops=64,
        restart_prob=0.3,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
        num_workers=1,
        dgl_graphs_file="/home/tandz/emojis/unsupervised_emojis/weibo/data/graph_bin/data.bin",
        num_samples=10000,
        num_copies=1,
        graph_transform=None,
        aug="rwr",
        num_neighbors=5,
        node_types=1,
        device='cpu'
    ):
        super(LoadBalanceGraphDataset).__init__()
        self.num_workers = num_workers
        self.rw_hops = rw_hops
        self.num_neighbors = num_neighbors
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        self.num_samples = num_samples
        self.node_types=node_types
        self.graphs, _ = dgl.data.utils.load_graphs(dgl_graphs_file)
        self.graphs = [g.to(device) for g in self.graphs]
        self.length = sum([g.number_of_nodes() for g in self.graphs])

        assert sum(step_dist) == 1.0
        assert positional_embedding_size > 1
        self.dgl_graphs_file = dgl_graphs_file
        graph_sizes = dgl.data.utils.load_labels(dgl_graphs_file)[
            "graph_sizes"
        ].tolist()
        print("load graph done")

        # a simple greedy algorithm for load balance
        # sorted graphs w.r.t its size in decreasing order
        # for each graph, assign it to the worker with least workload
        assert num_workers % num_copies == 0
        jobs = [list()] #[list() for i in range(num_workers // num_copies)]
        workloads = [0] #* (num_workers // num_copies)
        graph_sizes = sorted(
            enumerate(graph_sizes), key=operator.itemgetter(1), reverse=True
        )
        for idx, size in graph_sizes:
            argmin = workloads.index(min(workloads))
            workloads[argmin] += size
            jobs[argmin].append(idx)
        self.jobs = jobs * num_copies
        self.total = self.num_samples #* num_workers
        self.graph_transform = graph_transform
        assert aug in ("rwr", "ns")
        self.aug = aug

    def __len__(self):
        return self.num_samples # * self.num_workers

    def __iter__(self):
        degrees = torch.cat([g.in_degrees().double() ** 0.75 for g in self.graphs])
        prob = degrees / torch.sum(degrees)
        samples = np.random.choice(
            self.length, size=self.num_samples, replace=False, p=prob.cpu().numpy()
        )
        #print(len(samples), len(list(set(samples))))
        for idx in samples:
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            pos_n = torch.sum(self.graphs[i].ndata['label']).item()
            # max_len = pos_n if self.node_types==1 else self.graphs[i].number_of_nodes() - pos_n
            # start_idx = self.graphs[i].number_of_nodes() - pos_n if self.node_types==1 else 0
            
            if self.node_types==0:
                max_len = self.graphs[i].number_of_nodes() - pos_n
                start_idx = 0
            elif self.node_types==1:
                max_len = pos_n
                start_idx = self.graphs[i].number_of_nodes() - pos_n
            elif self.node_types==2:
                max_len = self.graphs[i].number_of_nodes()
                start_idx = 0
            else:
                raise Exception('node type unkonwn must be in (0,1,2)')


            # if node_idx < self.graphs[i].number_of_nodes():
            if node_idx < max_len:
                graph_idx = i
                node_idx = start_idx + node_idx
                break
            else:
                # node_idx -= self.graphs[i].number_of_nodes()
                node_idx = start_idx + node_idx % max_len 
        # print(idx, node_idx, start_idx, node_idx % max_len, max_len, pos_n)
        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = dgl.sampling.random_walk(
                g=self.graphs[graph_idx], nodes=[node_idx], length=step
            )[0][0][-1].item()
        # print("idx, other_node_idx:", idx, other_node_idx)
        if self.aug == "rwr":
            max_nodes_per_seed = max(
                self.rw_hops,
                int(
                    (
                        (self.graphs[graph_idx].out_degrees(node_idx) ** 0.75)
                        * math.e
                        / (math.e - 1)
                        / self.restart_prob
                    )
                    + 0.5
                ),
            )
            traces, _ = dgl.sampling.random_walk(
                self.graphs[graph_idx],
                nodes=[node_idx, other_node_idx],
                restart_prob=self.restart_prob,
                length=max_nodes_per_seed,
            )
        # print("traces", traces[0], traces[1])
        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        graph_k = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            positional_embedding_size=self.positional_embedding_size,
        )
        if self.graph_transform:
            graph_q = self.graph_transform(graph_q)
            graph_k = self.graph_transform(graph_k)
        return graph_q, graph_k


class LoadBalanceGraphDatasetInfer(torch.utils.data.IterableDataset):
    def __init__(
        self,
        rw_hops=64,
        restart_prob=0.3,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
        num_workers=1,
        dgl_graphs_file="/home/tandz/emojis/final_pj/another_task/unsupervised_emojis/data/graph_bin/data.bin",
        num_samples=10000,
        num_copies=1,
        graph_transform=None,
        aug="rwr",
        num_neighbors=5,
        node_types=1,
        device="cpu"
    ):
        super(LoadBalanceGraphDataset).__init__()
        self.num_workers = num_workers
        self.rw_hops = rw_hops
        self.num_neighbors = num_neighbors
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        self.num_samples = num_samples
        self.node_types=node_types
        self.graphs, _ = dgl.data.utils.load_graphs(dgl_graphs_file)
        self.graphs = [g.to(device) for g in self.graphs]
        self.length = sum([g.number_of_nodes() for g in self.graphs])

        assert sum(step_dist) == 1.0
        assert positional_embedding_size > 1
        self.dgl_graphs_file = dgl_graphs_file
        graph_sizes = dgl.data.utils.load_labels(dgl_graphs_file)[
            "graph_sizes"
        ].tolist()
        print("load graph done")

        # a simple greedy algorithm for load balance
        # sorted graphs w.r.t its size in decreasing order
        # for each graph, assign it to the worker with least workload
        assert num_workers % num_copies == 0
        jobs = [list()] #[list() for i in range(num_workers // num_copies)]
        workloads = [0] #* (num_workers // num_copies)
        graph_sizes = sorted(
            enumerate(graph_sizes), key=operator.itemgetter(1), reverse=True
        )
        for idx, size in graph_sizes:
            argmin = workloads.index(min(workloads))
            workloads[argmin] += size
            jobs[argmin].append(idx)
        self.jobs = jobs * num_copies
        self.total = self.num_samples #* num_workers
        self.graph_transform = graph_transform
        assert aug in ("rwr", "ns")
        self.aug = aug
        self.pos_n = torch.sum(self.graphs[0].ndata['label']).item()

    def __len__(self):
        return self.length-self.pos_n # * self.num_workers

    def __iter__(self):
        # degrees = torch.cat([g.in_degrees().double() ** 0.75 for g in self.graphs])
        # prob = degrees / torch.sum(degrees)
        # samples = np.random.choice(
        #     self.length, size=self.num_samples, replace=False, p=prob.numpy()
        # )
        #print(len(samples), len(list(set(samples))))
        for idx in range(self.length - self.pos_n):
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        # print("get:",idx)
        # for i in range(len(self.graphs)):
        #     pos_n = torch.sum(self.graphs[i].ndata['label']).item()
        #     max_len = pos_n if self.node_types==1 else self.graphs[i].number_of_nodes() - pos_n
        #     start_idx = self.graphs[i].number_of_nodes() - pos_n if self.node_types==1 else 0
        #     # if node_idx < self.graphs[i].number_of_nodes():
        #     if node_idx < max_len:
        #         graph_idx = i
        #         node_idx = start_idx + node_idx
        #         break
        #     else:
        #         # node_idx -= self.graphs[i].number_of_nodes()
        #         node_idx = start_idx + node_idx % max_len 
        # print(idx, node_idx, start_idx, node_idx % max_len, max_len, pos_n)
        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = dgl.sampling.random_walk(
                g=self.graphs[graph_idx], nodes=[node_idx], length=step
            )[0][0][-1].item()
        # print("idx, other_node_idx:", idx, other_node_idx)
        if self.aug == "rwr":
            max_nodes_per_seed = max(
                self.rw_hops,
                int(
                    (
                        (self.graphs[graph_idx].in_degrees(node_idx) ** 0.75)
                        * math.e
                        / (math.e - 1)
                        / self.restart_prob
                    )
                    + 0.5
                ),
            )
            traces, _ = dgl.sampling.random_walk(
                self.graphs[graph_idx],
                nodes=[node_idx, other_node_idx],
                restart_prob=self.restart_prob,
                length=max_nodes_per_seed,
            )
        # print("traces", traces[0], traces[1])
        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        graph_k = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            positional_embedding_size=self.positional_embedding_size,
        )
        if self.graph_transform:
            graph_q = self.graph_transform(graph_q)
            graph_k = self.graph_transform(graph_k)
        return graph_q, graph_k


class GraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rw_hops=64,
        subgraph_size=64,
        restart_prob=0.8,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
    ):
        super(GraphDataset).__init__()
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert sum(step_dist) == 1.0
        assert positional_embedding_size > 1
        #  graphs = []
        graphs, _ = dgl.data.utils.load_graphs(
            "data_bin/dgl/lscc_graphs.bin", [0, 1, 2]
        )
        for name in ["cs", "physics"]:
            g = Coauthor(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        for name in ["computers", "photo"]:
            g = AmazonCoBuy(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        # more graphs are comming ...
        print("load graph done")
        self.graphs = graphs
        self.length = sum([g.number_of_nodes() for g in self.graphs])

    def __len__(self):
        return self.length

    def _convert_idx(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()
        return graph_idx, node_idx

    def __getitem__(self, idx):
        graph_idx, node_idx = self._convert_idx(idx)

        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = dgl.contrib.sampling.random_walk(
                g=self.graphs[graph_idx], seeds=[node_idx], num_traces=1, num_hops=step
            )[0][0][-1].item()

        max_nodes_per_seed = max(
            self.rw_hops,
            int(
                (
                    self.graphs[graph_idx].out_degrees(node_idx)
                    * math.e
                    / (math.e - 1)
                    / self.restart_prob
                )
                + 0.5
            ),
        )
        # traces = dgl.contrib.sampling.random_walk_with_restart(
        #     self.graphs[graph_idx],
        #     seeds=[node_idx, other_node_idx],
        #     restart_prob=self.restart_prob,
        #     max_nodes_per_seed=max_nodes_per_seed,
        # )

        traces, _ = dgl.sampling.random_walk(
            self.graphs[graph_idx],
            nodes=[node_idx, other_node_idx],
            restart_prob=self.restart_prob,
            length=max_nodes_per_seed,
        )
        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )
        graph_k = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )
        return graph_q, graph_k


class NodeClassificationDataset(GraphDataset):
    def __init__(
        self,
        dgl_graphs_file="/home/tandz/emojis/unsupervised_emojis/data/graph_bin/data.bin",
        rw_hops=64,
        subgraph_size=64,
        restart_prob=0.5,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
        device='cpu'
    ):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert positional_embedding_size > 1

        # self.data = data_util.create_node_classification_dataset(dataset).data
        # get dgl graph
        self.graphs, _ = dgl.data.utils.load_graphs(dgl_graphs_file)
        self.graphs = [g.to(device) for g in self.graphs]
        # self.graphs = [self._create_dgl_graph(self.data)]
        self.length = sum([g.number_of_nodes() for g in self.graphs])
        self.total = self.length

    def _create_dgl_graph(self, data):
        graph = dgl.DGLGraph()
        src, dst = data.edge_index.tolist()
        num_nodes = data.edge_index.max() + 1
        graph.add_nodes(num_nodes)
        graph.add_edges(src, dst)
        graph.add_edges(dst, src)
        graph.readonly()
        return graph


class NodeClassificationDatasetLabeled(NodeClassificationDataset):
    def __init__(
        self,
        dataset,
        rw_hops=64,
        subgraph_size=64,
        restart_prob=0.8,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
        cat_prone=False,
    ):
        super(NodeClassificationDatasetLabeled, self).__init__(
            dataset,
            rw_hops,
            subgraph_size,
            restart_prob,
            positional_embedding_size,
            step_dist,
        )
        assert len(self.graphs) == 1
        self.num_classes = self.data.y.shape[1]

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()

        traces = dgl.contrib.sampling.random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=self.rw_hops,
        )

        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        return graph_q, self.data.y[idx].argmax().item()


def GetLinkPredictDataloader(batch_size, conv_layers, data, split=None):
    train, test = data
    g, pos_g, neg_g = train
    nids = [i for i in range(g.number_of_nodes())]
    if split:
        random.shuffle(nids)
        train_nids = nids[:int(len(nids)*0.9)]
    else:
        train_nids =nids
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(conv_layers)
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(5))

    dataloader = dgl.dataloading.DataLoader(
        g, train_nids, sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        # pin_memory=True,
        num_workers=4)

    return dataloader

def GetLinkPredictGraph(load_path="/home/tandz/emojis/unsupervised_emojis/data/", device="cpu"):
    train_data = torch.load(load_path + "train.t")
    test_data = torch.load(load_path + "test.t")

    # train_g, train_pos_g, train_neg_g
    train = parse_data_2_graph(train_data, device)
    test = parse_data_2_graph(test_data, device)

    return train, test


def parse_data_2_graph(data, device="cpu"):
    data.to(device)
    g = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes)
    g.ndata["emb"] = data.x
    g.ndata["label"] = data.y
    g.ndata["nindex"] = torch.range(0,data.num_nodes-1, device=device)
    pos_uv = data.edge_label_index[:,torch.where(data.edge_label==1)[0]]
    neg_uv = data.edge_label_index[:,torch.where(data.edge_label==0)[0]]

    pos_g = dgl.graph((pos_uv[0], pos_uv[1]), num_nodes=data.num_nodes)
    neg_g = dgl.graph((neg_uv[0], neg_uv[1]), num_nodes=data.num_nodes)

    return g, pos_g, neg_g


if __name__ == "__main__":
    num_workers = 1
    import psutil

    mem = psutil.virtual_memory()
    print(mem.used / 1024 ** 3)
    graph_dataset = LoadBalanceGraphDataset(
        num_workers=num_workers, aug="rwr", rw_hops=4, num_neighbors=5
    )
    mem = psutil.virtual_memory()
    print(mem.used / 1024 ** 3)
    graph_loader = torch.utils.data.DataLoader(
        graph_dataset,
        batch_size=1,
        collate_fn=data_util.batcher(),
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )
    mem = psutil.virtual_memory()
    print(mem.used / 1024 ** 3)
    for step, batch in enumerate(graph_loader):
        # print(torch.device(0))
        print(batch[0].to(torch.device(0)))
        print(batch[0].device)
        print("bs", batch[0].batch_size)
        print("n=", batch[0].number_of_nodes())
        print("m=", batch[0].number_of_edges())
        # mem = psutil.virtual_memory()
        # print(mem.used / 1024 ** 3)
        # print(batch[0].ndata["pos_undirected"])
        print("label", batch[0].ndata["label"])
        print("nindex", batch[0].ndata["nindex"])
        print(batch[0].nodes())
        print(batch[0].edges())
        print("label", batch[1].ndata["label"])
        print("nindex", batch[1].ndata["nindex"])
        print(batch[1].nodes())
        print(batch[1].edges())
        break
    exit(0)
