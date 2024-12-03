
import math
import operator

import dgl
import dgl.data
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from dgl.data import AmazonCoBuy, Coauthor
try:
    import gcc.datasets.data_util as data_util
except:
    import data_util as data_util
import random
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.graphs, _ = dgl.data.utils.load_graphs(
        dataset.dgl_graphs_file, dataset.jobs[worker_id]
    )
    dataset.length = sum([g.number_of_nodes(dataset.etype[0]) for g in dataset.graphs])
    # for g in dataset.graphs:
    #     g.create_formats_()
        
    np.random.seed(worker_info.seed % (2 ** 32))

def merge_nlist(ls:list):
    res = []
    for l in ls:
        res.extend(l)
    return res

class LoadBalanceGraphDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        rw_hops=64,
        restart_prob=0.3,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
        num_workers=1,
        dgl_graphs_file="/home/tandz/emojis/unsupervised_emojis/data/graph_bin/data.bin",
        num_samples=10000,
        num_copies=1,
        graph_transform=None,
        aug="rwr",
        num_neighbors=5,
        node_types=1,
        device='cpu',
        etype=('post','hase','emoji'),
        metapath=['hase','ein']
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
        self.device = device
        self.graphs = [g.to(device) for g in self.graphs]
        self.length = sum([g.num_nodes(etype[0]) for g in self.graphs])
        self.etype = etype
        self.metapath = metapath
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
        degrees = torch.cat([g.out_degrees(etype=self.etype).double() ** 0.75 for g in self.graphs])
        degrees = torch.where(degrees<=1, torch.zeros_like(degrees), degrees)
        prob = degrees / torch.sum(degrees)
        # print(self.length, prob.size())
        samples = np.random.choice(
            self.length, size=self.num_samples, replace=True, p=prob.cpu().numpy()
        )
        # print(len(samples), len(list(set(samples))))
        for idx in samples:
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
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
                        (self.graphs[graph_idx].out_degrees(etype=self.etype)[node_idx] ** 0.75)
                        * math.e
                        / (math.e - 1)
                        / self.restart_prob
                    )
                    + 0.5
                ),
            )
            max_nodes_per_seed = min(max_nodes_per_seed, 100)
            traces, eids = dgl.sampling.random_walk(
                self.graphs[graph_idx],
                metapath=self.metapath*int(max_nodes_per_seed//2),
                nodes=[node_idx, other_node_idx],
                # restart_prob=torch.tensor([0.0, 0.0]*int(max_nodes_per_seed//2), device=self.device),#self.restart_prob,
            )
        # print("traces", traces, eids, max_nodes_per_seed)
        graph_q = data_util._rwr_trace_to_dgl_hetegraph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            eids=eids,
            positional_embedding_size=self.positional_embedding_size,
        )
        graph_k = data_util._rwr_trace_to_dgl_hetegraph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            eids=eids,
            positional_embedding_size=self.positional_embedding_size,
        )
        if self.graph_transform:
            graph_q = self.graph_transform(graph_q)
            graph_k = self.graph_transform(graph_k)

        try:
            graph_q = dgl.to_homogeneous(graph_q, ['feat', 'seed'])
            graph_k = dgl.to_homogeneous(graph_k, ['feat', 'seed'])
        except:
            print("error")
            raise EnvironmentError

        return graph_q, graph_k

class GraphCLDataset(LoadBalanceGraphDataset):
    def __init__(
        self,
        rw_hops=64,
        restart_prob=0.3,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
        num_workers=1,
        dgl_graphs_file="/home/tandz/emojis/unsupervised_emojis/xhs/data/graph_bin/train_g4.bin",
        num_samples=10000,
        num_copies=1,
        graph_transform=None,
        aug="rwr",
        num_neighbors=5,
        node_types=1,
        device='cpu',
        etype=('post','hase','emoji'),
        metapath=['hase','ein']
        ):
        super().__init__(
            rw_hops,
            restart_prob,
            positional_embedding_size,
            step_dist,
            num_workers,
            dgl_graphs_file,
            num_samples,
            num_copies,
            graph_transform,
            aug,
            num_neighbors,
            node_types,
            device,
            etype,
            metapath
        )

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        other_node_idx = node_idx
        max_nodes_per_seed = max(
            self.rw_hops,
            int(
                (
                    (self.graphs[graph_idx].out_degrees(etype=self.etype)[node_idx] ** 0.75)
                    * math.e
                    / (math.e - 1)
                    / self.restart_prob
                )
                + 0.5
            ),
        )
        max_nodes_per_seed = min(max_nodes_per_seed, 100)
        
        node_idxs = [node_idx]*int(max_nodes_per_seed//len(self.metapath))
        other_node_idxs = [other_node_idx]*int(max_nodes_per_seed//len(self.metapath))

        traces, eids = dgl.sampling.random_walk(
            self.graphs[graph_idx],
            metapath=self.metapath,
            nodes=node_idxs+other_node_idxs,
            restart_prob=torch.tensor([0.0, 0.1], device=self.device)#*int(max_nodes_per_seed//2), device=self.device),#self.restart_prob,
        )
        # print("traces", traces, eids, max_nodes_per_seed)
        graph_q = data_util._rwr_trace_to_dgl_hetegraph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=merge_nlist(traces[:len(node_idxs)]),
            eids=torch.tile(eids,(1,len(node_idxs)))[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        graph_k = data_util._rwr_trace_to_dgl_hetegraph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=merge_nlist(traces[len(node_idxs):]),
            eids=torch.tile(eids,(1,len(node_idxs)))[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        if self.graph_transform:
            graph_q = self.graph_transform(graph_q)
            graph_k = self.graph_transform(graph_k)

        try:
            graph_q = dgl.to_homogeneous(graph_q, ['feat', 'seed'])
            graph_k = dgl.to_homogeneous(graph_k, ['feat', 'seed'])
        except:
            print("error")
            raise EnvironmentError

        return graph_q, graph_k

class GraphCLDataset_NS(LoadBalanceGraphDataset):
    def __init__(
        self,
        rw_hops=64,
        restart_prob=0.3,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
        num_workers=1,
        dgl_graphs_file="/home/tandz/emojis/unsupervised_emojis/xhs/data/graph_bin/train_g4.bin",
        num_samples=10000,
        num_copies=1,
        graph_transform=None,
        aug="rwr",
        num_neighbors=5,
        node_types=1,
        device='cpu',
        etype=('post','hase','emoji'),
        metapath=['hase','ein']
        ):
        super().__init__(
            rw_hops,
            restart_prob,
            positional_embedding_size,
            step_dist,
            num_workers,
            dgl_graphs_file,
            num_samples,
            num_copies,
            graph_transform,
            aug,
            num_neighbors,
            node_types,
            device,
            etype,
            metapath
        )

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        other_node_idx = node_idx
        max_nodes_per_seed = max(
            self.rw_hops,
            int(
                (
                    (self.graphs[graph_idx].out_degrees(etype=self.etype)[node_idx] ** 0.75)
                    * math.e
                    / (math.e - 1)
                    / self.restart_prob
                )
                + 0.5
            ),
        )
        max_nodes_per_seed = min(max_nodes_per_seed, 100)
        
        node_idxs = [node_idx]*int(max_nodes_per_seed//len(self.metapath))
        other_node_idxs = [other_node_idx]*int(max_nodes_per_seed//len(self.metapath))

        graph_q = dgl.sampling.sample_neighbors(
            self.graphs[graph_idx],
            metapath=self.metapath,
            nodes=node_idxs+other_node_idxs,
            restart_prob=torch.tensor([0.0, 0.1], device=self.device)#*int(max_nodes_per_seed//2), device=self.device),#self.restart_prob,
        )
        graph_k = dgl.sampling.sample_neighbors(
            self.graphs[graph_idx],
            metapath=self.metapath,
            nodes=node_idxs+other_node_idxs,
            restart_prob=torch.tensor([0.0, 0.1], device=self.device)#*int(max_nodes_per_seed//2), device=self.device),#self.restart_prob,
        )

        if self.graph_transform:
            graph_q = self.graph_transform(graph_q)
            graph_k = self.graph_transform(graph_k)

        try:
            graph_q = dgl.to_homogeneous(graph_q, ['feat', 'seed'])
            graph_k = dgl.to_homogeneous(graph_k, ['feat', 'seed'])
        except:
            print("error")
            raise EnvironmentError

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
                    self.graphs[graph_idx].out_degrees(node_idx, etype=self.etype)
                    * math.e
                    / (math.e - 1)
                    / self.restart_prob
                )
                + 0.5
            ),
        )
        max_nodes_per_seed = min(max_nodes_per_seed, 32)
        # traces = dgl.contrib.sampling.random_walk_with_restart(
        #     self.graphs[graph_idx],
        #     seeds=[node_idx, other_node_idx],
        #     restart_prob=self.restart_prob,
        #     max_nodes_per_seed=max_nodes_per_seed,
        # )

        traces, eids = dgl.sampling.random_walk(
            self.graphs[graph_idx],
            metapath=self.metapath*int(max_nodes_per_seed//2),
            nodes=[node_idx, other_node_idx],
            # nodes=[node_idx, other_node_idx],
            # restart_prob=self.restart_prob,
            # length=max_nodes_per_seed,
        )
        graph_q = data_util._rwr_trace_to_dgl_hetegraph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            eids=eids,
            positional_embedding_size=self.positional_embedding_size,
            # entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )
        graph_k = data_util._rwr_trace_to_dgl_hetegraph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            eids=eids,
            positional_embedding_size=self.positional_embedding_size,
            # entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )


        try:
            graph_q = dgl.to_homogeneous(graph_q, ['feat', 'seed'])
            graph_k = dgl.to_homogeneous(graph_k, ['feat', 'seed'])
        except:
            # single nodes graph
            traces, eids = dgl.sampling.random_walk(
                self.graphs[graph_idx],
                metapath=['win','hase','by']*(max_nodes_per_seed//3),
                nodes=[node_idx, other_node_idx],
                # nodes=[node_idx, other_node_idx],
                # restart_prob=self.restart_prob,
                # length=max_nodes_per_seed,
            )
            graph_q = data_util._rwr_trace_to_dgl_hetegraph(
                g=self.graphs[graph_idx],
                seed=node_idx,
                trace=traces[0],
                eids=eids,
                positional_embedding_size=self.positional_embedding_size,
                # entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
            )
            graph_k = data_util._rwr_trace_to_dgl_hetegraph(
                g=self.graphs[graph_idx],
                seed=other_node_idx,
                trace=traces[1],
                eids=eids,
                positional_embedding_size=self.positional_embedding_size,
                # entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
            )
            graph_q = dgl.to_homogeneous(graph_q, ['feat', 'seed'])
            graph_k = dgl.to_homogeneous(graph_k, ['feat', 'seed'])

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
        device='cpu',
        etype=('post','hase','emoji'),
        metapath=['hase','ein']
    ):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        self.etype = etype
        self.metapath = metapath
        assert positional_embedding_size > 1
        self.device=device
        # get dgl graph
        self.graphs, _ = dgl.data.utils.load_graphs(dgl_graphs_file)
        self.graphs = [g.to(device) for g in self.graphs]
        self.length = sum([g.num_nodes(etype[0]) for g in self.graphs])
        self.total = self.length

class NodeClassificationDataset2(NodeClassificationDataset):
    def __init__(
        self,
        dgl_graphs_file="/home/tandz/emojis/unsupervised_emojis/data/graph_bin/data.bin",
        rw_hops=64,
        subgraph_size=64,
        restart_prob=0.5,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
        device='cpu',
        etype=('post','hase','emoji'),
        metapath=['hase','ein']
    ):
        super().__init__(
            dgl_graphs_file,
            rw_hops,
            subgraph_size,
            restart_prob,
            positional_embedding_size,
            step_dist,
            device,
            etype,
            metapath
        )

    def __getitem__(self, idx):
        graph_idx, node_idx = self._convert_idx(idx)
        other_node_idx = node_idx

        max_nodes_per_seed = max(
            self.rw_hops,
            int(
                (
                    self.graphs[graph_idx].out_degrees(node_idx, etype=self.etype)
                    * math.e
                    / (math.e - 1)
                    / self.restart_prob
                )
                + 0.5
            ),
        )
        max_nodes_per_seed = min(max_nodes_per_seed, 100)

        node_idxs = [node_idx]*int(max_nodes_per_seed//len(self.metapath))
        other_node_idxs = [other_node_idx]*int(max_nodes_per_seed//len(self.metapath))

        traces, eids = dgl.sampling.random_walk(
            self.graphs[graph_idx],
            metapath=self.metapath,
            nodes=node_idxs+other_node_idxs,
            restart_prob=torch.tensor([0.0, 0.1], device=self.device)#*int(max_nodes_per_seed//2), device=self.device),#self.restart_prob,
        )
        # print("traces", traces, eids, max_nodes_per_seed)
        graph_q = data_util._rwr_trace_to_dgl_hetegraph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=merge_nlist(traces[:len(node_idxs)]),
            eids=torch.tile(eids,(1,len(node_idxs)))[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        graph_k = data_util._rwr_trace_to_dgl_hetegraph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=merge_nlist(traces[len(node_idxs):]),
            eids=torch.tile(eids,(1,len(other_node_idxs)))[0],
            positional_embedding_size=self.positional_embedding_size,
        )

        try:
            graph_q = dgl.to_homogeneous(graph_q, ['feat', 'seed'])
            graph_k = dgl.to_homogeneous(graph_k, ['feat', 'seed'])
        except:
            # single nodes graph
            traces, eids = dgl.sampling.random_walk(
                self.graphs[graph_idx],
                metapath=['win','hase'],
                nodes=node_idxs+other_node_idxs,
                restart_prob=torch.tensor([0.0, 0.1], device=self.device)
            )
            graph_q = data_util._rwr_trace_to_dgl_hetegraph(
                g=self.graphs[graph_idx],
                seed=node_idx,
                trace=merge_nlist(traces[:len(node_idxs)]),
                eids=torch.tile(eids,(1,len(node_idxs)))[0],
                positional_embedding_size=self.positional_embedding_size,
            )
            graph_k = data_util._rwr_trace_to_dgl_hetegraph(
                g=self.graphs[graph_idx],
                seed=other_node_idx,
                trace=merge_nlist(traces[len(node_idxs):]),
                eids=torch.tile(eids,(1,len(other_node_idx)))[0],
                positional_embedding_size=self.positional_embedding_size,
            )
            graph_q = dgl.to_homogeneous(graph_q, ['feat', 'seed'])
            graph_k = dgl.to_homogeneous(graph_k, ['feat', 'seed'])

        return graph_q, graph_k

class NodeClassificationDataset3(NodeClassificationDataset):
    def __init__(
        self,
        dgl_graphs_file="/home/tandz/emojis/unsupervised_emojis/data/graph_bin/data.bin",
        rw_hops=64,
        subgraph_size=64,
        restart_prob=0.5,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
        device='cpu',
        etype=('post','hase','emoji'),
        metapath=['hase','ein']
    ):
        super().__init__(
            dgl_graphs_file,
            rw_hops,
            subgraph_size,
            restart_prob,
            positional_embedding_size,
            step_dist,
            device,
            etype,
            metapath
        )
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)

    def __getitem__(self, idx):
        graph_idx, node_idx = self._convert_idx(idx)
        other_node_idx = node_idx
        ntype = self.etype[0]
        subg = self.graphs[0].subgraph(self.sampler.sample(self.graphs[0], {ntype:[node_idx]})[0])
        ntypes = subg.ntypes
        nnodes = [subg.num_nodes(n) for n in ntypes]
        seed={}
        for i,k in enumerate(ntypes):
            seed[k] = torch.zeros(nnodes[i], dtype=torch.long).to(subg.device)
        subg.ndata.update({'seed':seed})
        try:
            subg.ndata["seed"][ntype][0] = 1
        except:
            print("Error: Subgraph's len < 1 =:",len(subg.ndata["seed"][self.etype[0]]),self.etype,subg)
        graph_q = dgl.to_homogeneous(subg, ['feat', 'seed'])
        graph_k = graph_q
        return graph_q, graph_k



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

def parse_lp_dataset():
    # Split edge set for training and testing
    g=dgl.data.utils.load_graphs("/home/tandz/emojis/unsupervised_emojis/xhs/data/graph_bin/train_g.bin")
    u, v = g.edges()
    eids = np.arange(g.num_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = g.num_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.num_edges())
    test_neg_u, test_neg_v = (
        neg_u[neg_eids[:test_size]],
        neg_v[neg_eids[:test_size]],
    )
    train_neg_u, train_neg_v = (
        neg_u[neg_eids[test_size:]],
        neg_v[neg_eids[test_size:]],
    )



class LoadBalanceGraphDatasetLP2(torch.utils.data.IterableDataset):
    def __init__(
        self,
        rw_hops=64,
        restart_prob=0.3,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
        num_workers=1,
        dgl_graphs_file="/home/tandz/emojis/unsupervised_emojis/data/graph_bin/data.bin",
        num_samples=10000,
        num_copies=1,
        graph_transform=None,
        aug="rwr",
        num_neighbors=5,
        node_types=1,
        device='cpu',
        etype=('post','hase','emoji'),
        metapath=['hase','ein']
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
        self.device = device
        self.graphs = [g.to(device) for g in self.graphs]
        self.length = sum([g.num_nodes(etype[0]) for g in self.graphs])
        self.etype = etype
        self.metapath = metapath
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
        degrees = torch.cat([g.out_degrees(etype=self.etype).double() ** 0.75 for g in self.graphs])
        prob = degrees / torch.sum(degrees)
        samples = np.random.choice(
            self.length, size=self.num_samples, replace=False, p=prob.cpu().numpy()
        )
        for idx in samples:
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        if self.aug == "rwr":
            max_nodes_per_seed = max(
                self.rw_hops,
                int(
                    (
                        (self.graphs[graph_idx].out_degrees(etype=self.etype)[node_idx] ** 0.75)
                        * math.e
                        / (math.e - 1)
                        / self.restart_prob
                    )
                    + 0.5
                ),
            )
            traces, eids = dgl.sampling.random_walk(
                self.graphs[graph_idx],
                metapath=self.metapath*int(max_nodes_per_seed//2),
                nodes=[node_idx],
            )
        pos_g = data_util._rwr_trace_to_dgl_hetegraph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            eids=eids,
            positional_embedding_size=self.positional_embedding_size,
        )
        if self.graph_transform:
            pos_g = self.graph_transform(pos_g)
        
        neg_g = construct_negative_graph(pos_g, 5, self.etype)

        try:
            neg_g = dgl.to_homogeneous(neg_g, ['feat', 'seed'])
            pos_g = dgl.to_homogeneous(pos_g, ['feat', 'seed'])
        except:
            print("error")
            raise EnvironmentError
        
        return pos_g, neg_g

class LoadBalanceGraphDatasetLP(torch.utils.data.IterableDataset):
    def __init__(
        self,
        rw_hops=64,
        restart_prob=0.3,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
        num_workers=1,
        dgl_graphs_file="/home/tandz/emojis/unsupervised_emojis/data/graph_bin/data.bin",
        num_samples=10000,
        num_copies=1,
        graph_transform=None,
        aug="rwr",
        num_neighbors=5,
        node_types=1,
        device='cpu',
        etype=('post','hase','emoji'),
        metapath=['hase','ein']
    ):
        super(LoadBalanceGraphDatasetLP).__init__()
        self.num_workers = num_workers
        self.rw_hops = rw_hops
        self.num_neighbors = num_neighbors
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        self.num_samples = num_samples
        self.node_types=node_types
        self.graphs, _ = dgl.data.utils.load_graphs(dgl_graphs_file)
        self.device = device
        self.graphs = [g.to(device) for g in self.graphs]
        self.length = sum([g.num_nodes(etype[0]) for g in self.graphs])
        self.etype = etype
        self.metapath = metapath
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
        degrees = torch.cat([g.out_degrees(etype=self.etype).double() ** 0.75 for g in self.graphs])
        prob = degrees / torch.sum(degrees)
        # print(self.length, prob.size())
        samples = np.random.choice(
            self.length, size=self.num_samples, replace=True, p=prob.cpu().numpy()
        )
        # print(len(samples), len(list(set(samples))))
        for idx in samples:
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
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
                        (self.graphs[graph_idx].out_degrees(etype=self.etype)[node_idx] ** 0.75)
                        * math.e
                        / (math.e - 1)
                        / self.restart_prob
                    )
                    + 0.5
                ),
            )
            traces, eids = dgl.sampling.random_walk(
                self.graphs[graph_idx],
                metapath=self.metapath*int(max_nodes_per_seed//len(self.metapath)),
                nodes=[node_idx, other_node_idx],
                # restart_prob=torch.tensor([0.0, 0.0]*int(max_nodes_per_seed//2), device=self.device),#self.restart_prob,
            )
        # print("traces", traces, eids, max_nodes_per_seed)
        graph_q = data_util._rwr_trace_to_dgl_hetegraph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            eids=eids,
            positional_embedding_size=self.positional_embedding_size,
        )
        graph_k = data_util._rwr_trace_to_dgl_hetegraph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            eids=eids,
            positional_embedding_size=self.positional_embedding_size,
        )
        if self.graph_transform:
            graph_q = self.graph_transform(graph_q)
            graph_k = self.graph_transform(graph_k)

        neg_g = construct_negative_graph(graph_q, 5, self.etype)

        try:
            graph_q = dgl.to_homogeneous(graph_q, ['feat', 'seed'])
            graph_k = dgl.to_homogeneous(graph_k, ['feat', 'seed'])
            neg_g = dgl.to_homogeneous(neg_g)
        except:
            print("error")
            raise EnvironmentError

        return graph_q, graph_k, neg_g
   
def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,), device=graph.device)
    neg_g =  dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}
        )
    # neg_g.ndata.update(graph.ndata)
    return neg_g

def get_linkpred_batch_loader(graph, 
                                batch_size=1024,
                                num_layers=2, 
                                device='cpu'):      
    negative_sampler = dgl.dataloading.negative_sampler.Uniform(1)
    sampler = dgl.dataloading.NeighborSampler([-1])
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, negative_sampler=negative_sampler
    )
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DataLoader.
        graph,  # The graph
        torch.arange(graph.num_edges(), device=device),  # The edges to iterate over
        sampler,  # The neighbor sampler
        device=device,  # Put the MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=batch_size,  # Batch size
        shuffle=True,  # Whether to shuffle the nodes for every epoch
        drop_last=False,  # Whether to drop the last incomplete batch
        num_workers=0,  # Number of sampler processes
    )
    return train_dataloader
    # input_nodes, pos_graph, neg_graph, mfgs = next(iter(train_dataloader))
    # print(input_nodes, pos_graph, neg_graph, mfgs)

class NegativeSampler(object):
    def __init__(self, g, k):
        # caches the probability distribution
        self.weights = {
            etype: g.in_degrees(etype=etype).float() ** 0.75
            for etype in g.canonical_etypes}
        self.k = k

    def __call__(self, g, eids_dict):
        result_dict = {}
        for etype, eids in eids_dict.items():
            src, _ = g.find_edges(eids, etype=etype)
            src = src.repeat_interleave(self.k)
            dst = self.weights[etype].multinomial(len(src), replacement=True)
            result_dict[etype] = (src, dst)
        return result_dict
    
def get_hete_linkpred_batch_loader(graph,
                                etype, 
                                batch_size=1,
                                num_layers=1,
                                device='cpu', shuffle=True, neg=1):
    g = graph
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(neg))
    
    # sampler = dgl.dataloading.as_edge_prediction_sampler(
    #     sampler, negative_sampler=NegativeSampler(g, 1))

    train_eid_dict = {
        etype: g.edges(etype=etype, form='eid')
        # for etype in g.canonical_etypes
        }
    
    dataloader = dgl.dataloading.DataLoader(
        g, train_eid_dict, sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,device=device)
    
    return dataloader
    # input_nodes, pos_graph, neg_graph, mfgs = next(iter(dataloader))
    # print(input_nodes, pos_graph, neg_graph, mfgs)


class LoadLPHeteGraphDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        etype=('post','hase','emoji'), # canonical_etypes
        num_workers=1,
        dgl_graphs_file="/home/tandz/emojis/unsupervised_emojis/xhs/data/graph_bin/train_hg3.bin",
        device='cpu',
        batch_size=1024,
        num_samples=10000
    ):
        super(LoadBalanceGraphDataset).__init__()
        self.etype = etype
        self.num_workers = num_workers
        self.graphs, _ = dgl.data.utils.load_graphs(dgl_graphs_file)
        self.device = device
        self.graphs = [g.to(device) for g in self.graphs]
        self.batch_size = batch_size
        self.dgl_graphs_file = dgl_graphs_file
        graph_sizes = dgl.data.utils.load_labels(dgl_graphs_file)[
            "graph_sizes"
        ].tolist()
        print("load graph done")

        jobs = [list()] #[list() for i in range(num_workers // num_copies)]
        workloads = [0] #* (num_workers // num_copies)
        graph_sizes = sorted(
            enumerate(graph_sizes), key=operator.itemgetter(1), reverse=True
        )
        for idx, size in graph_sizes:
            argmin = workloads.index(min(workloads))
            workloads[argmin] += size
            jobs[argmin].append(idx)
        self.dloader = self.load_lp_dataloader()
        self.total = self.dloader.__len__()
        self.num_samples = num_samples
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)

    def load_lp_dataloader(self):
        return get_hete_linkpred_batch_loader(graph=self.graphs[0],etype=self.etype,batch_size=self.batch_size, device=self.device, num_layers=1)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for i in range(int(self.num_samples/self.batch_size)):
            yield self.__getitem__(next(iter(self.dloader)))

    def __getitem__(self, data):
        input_nodes, pos_graph, neg_graph, mfgs = data
        subgs=[]
        # for ntype in pos_graph.ntypes:
        #     for nid in pos_graph.ndata['_ID'][ntype]:
        #         subgs.append(dgl.subgraph(self.graphs[0], self.sampler.sample(self.graphs[0], {ntype:[nid]})[0]))
        for ntype in pos_graph.ndata['_ID']:
            for nid in pos_graph.ndata['_ID'][ntype]:
                subg = self.graphs[0].subgraph(self.sampler.sample(self.graphs[0], {ntype:[nid.item()]})[0])
                # seed = torch.zeros(subg.num_nodes(), dtype=torch.long).to(subg.device)
                # seed[0] = 1
                ntypes = subg.ntypes
                nnodes = [subg.num_nodes(n) for n in ntypes]
                seed={}
                for i,k in enumerate(ntypes):
                    seed[k] = torch.zeros(nnodes[i], dtype=torch.long).to(subg.device)
                subg.ndata.update({'seed':seed})
                try:
                    subg.ndata["seed"][ntype][0] = 1
                except:
                    print("Error: Subgraph's len < 1 =:",len(subg.ndata["seed"][self.etype[0]]),self.etype,subg)
                subgs.append(dgl.to_homogeneous(subg, ['feat', 'seed']))
        return  dgl.to_homogeneous(pos_graph), dgl.to_homogeneous(neg_graph), dgl.batch(subgs)

class LoadLPHeteGraphDataset2(LoadLPHeteGraphDataset):
    def __init__(
        self,
        etype=('post','hase','emoji'), # canonical_etypes
        num_workers=1,
        dgl_graphs_file="/home/tandz/emojis/unsupervised_emojis/xhs/data/graph_bin/train_hg3.bin",
        device='cpu',
        batch_size=1024,
        num_samples=10000,
        metapath=['hase','ein']
    ):  
        super().__init__(
            etype,
            num_workers,
            dgl_graphs_file,
            device,
            batch_size,
            num_samples,
            metapath
        )
        edge_fanout = {
            'by': 0, 
            'ein': 0, 
            'with': 0, 
            'hase': 0, 
            'hasw': 0, 
            'win': 0, 
            'withe': 0
        }
        fan_outs=[]
        for edge in self.metapath:
            fan_outs.append(edge_fanout.copy()[edge].update({edge:-1}))
        self.sampler = dgl.dataloading.NeighborSampler(fan_outs)

class LoadLPGraphDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        num_workers=1,
        dgl_graphs_file="/home/tandz/emojis/unsupervised_emojis/xhs/data/graph_bin/train_hg3.bin",
        device='cpu',
        batch_size=1024,
        num_samples=10000,
    ):
        super(LoadBalanceGraphDataset).__init__()
        self.num_workers = num_workers
        self.graphs, _ = dgl.data.utils.load_graphs(dgl_graphs_file)
        self.device = device
        self.graphs = [g.to(device) for g in self.graphs]
        self.batch_size = batch_size
        self.dgl_graphs_file = dgl_graphs_file
        graph_sizes = dgl.data.utils.load_labels(dgl_graphs_file)[
            "graph_sizes"
        ].tolist()
        print("load graph done")

        jobs = [list()] #[list() for i in range(num_workers // num_copies)]
        workloads = [0] #* (num_workers // num_copies)
        graph_sizes = sorted(
            enumerate(graph_sizes), key=operator.itemgetter(1), reverse=True
        )
        for idx, size in graph_sizes:
            argmin = workloads.index(min(workloads))
            workloads[argmin] += size
            jobs[argmin].append(idx)
        self.dloader = self.load_lp_dataloader()
        self.total = self.dloader.__len__()
        self.num_samples = num_samples
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)

    def load_lp_dataloader(self):
        return get_linkpred_batch_loader(graph=self.graphs[0],batch_size=self.batch_size, device=self.device, num_layers=1)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for i in range(int(self.num_samples/self.batch_size)):
            yield self.__getitem__(next(iter(self.dloader)))

    def __getitem__(self, data):
        input_nodes, pos_graph, neg_graph, mfgs = data
        subgs=[]
        # for ntype in pos_graph.ntypes:
        #     for nid in pos_graph.ndata['_ID'][ntype]:
        #         subgs.append(dgl.subgraph(self.graphs[0], self.sampler.sample(self.graphs[0], {ntype:[nid]})[0]))
        for nid in pos_graph.ndata['_ID']:
            subg = self.graphs[0].subgraph(self.sampler.sample(self.graphs[0], [nid.item()])[0])
            seed = torch.zeros(subg.num_nodes(), dtype=torch.long).to(subg.device)
            seed[0] = 1
            subg.ndata.update({'seed': seed})
            subgs.append(subg)
        return pos_graph, neg_graph, dgl.batch(subgs)

if __name__ == "__main__":
    num_workers = 1
    import psutil

    mem = psutil.virtual_memory()
    print(mem.used / 1024 ** 3)
    # graph_dataset = LoadLPHeteGraphDataset(dgl_graphs_file="/home/tandz/emojis/unsupervised_emojis/xhs/data/graph_bin/train_g3.bin", 
    #                                        etype='win', batch_size=12, device='cuda:1')
    # graph_dataset = GraphCLDataset()
    # mem = psutil.virtual_memory()
    # print(mem.used / 1024 ** 3)
    # graph_loader = torch.utils.data.DataLoader(
    #     graph_dataset,
    #     batch_size=2,
    #     collate_fn=data_util.batcher(),#batcher_pl(),
    #     # num_workers=num_workers,
    #     # worker_init_fn=worker_init_fn,
    # )
    # mem = psutil.virtual_memory()
    # print(mem.used / 1024 ** 3)
    # for step, batch in enumerate(graph_loader):
    #     # print(torch.device(0))
    #     print(step, batch)
    #     # print(batch[0].to(torch.device(0)))
    #     # print(batch[0].device)
    #     # print("bs", batch[0].batch_size)
    #     # print("n=", batch[0].number_of_nodes())
    #     # print("m=", batch[0].number_of_edges())
    #     # # mem = psutil.virtual_memory()
    #     # # print(mem.used / 1024 ** 3)
    #     # # print(batch[0].ndata["pos_undirected"])
    #     # print("label", batch[0].ndata["seed"])
    #     # # print(batch[0].nodes())
    #     # # print(batch[0].edges())
    #     # print("label", batch[1].ndata["seed"])
    #     # # print(batch[1].nodes())
    #     # # print(batch[1].edges())
    #     break
    
    # get_hete_linkpred_batch_loader(dgl.data.utils.load_graphs("/home/tandz/emojis/unsupervised_emojis/xhs/data/graph_bin/train_g3.bin")[0][0], 
    #                                etype='win', batch_size=16, device='cuda:1', num_layers=1)

    # edge_fanout = {
    #     ('emoji', 'by', 'word'): 0, 
    #     ('emoji', 'ein', 'post'): 0, 
    #     ('emoji', 'with', 'emoji'): 0, 
    #     ('post', 'hase', 'emoji'): 0, 
    #     ('post', 'hasw', 'word'): 0, 
    #     ('word', 'win', 'post'): 0, 
    #     ('word', 'withe', 'emoji'): 0
    # }
    # edge_fanout[('emoji', 'with', 'emoji')] = 3
    edge_fanout = {
        'by': 0, 
        'ein': 0, 
        'with': 0, 
        'hase': 0, 
        'hasw': 0, 
        'win': 0, 
        'withe': 0
    }
    edge_fanout['by'] = -1
    g=dgl.data.utils.load_graphs("/home/tandz/emojis/unsupervised_emojis/xhs/data/graph_bin/small_test.bin")[0][0]
    graph_q = dgl.sampling.sample_neighbors(
        g,
        {'emoji':0},
        edge_fanout,
        edge_dir='out'
    )


    exit(0)
