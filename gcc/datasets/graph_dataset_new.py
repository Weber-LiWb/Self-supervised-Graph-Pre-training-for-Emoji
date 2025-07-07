
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
    from . import data_util
except ImportError:
    try:
        import gcc.datasets.data_util as data_util
    except ImportError:
        import data_util
import random
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.graphs, _ = dgl.data.utils.load_graphs(
        dataset.dgl_graphs_file, dataset.jobs[worker_id]
    )
    dataset.length = sum([g.number_of_nodes(dataset.etype[-1]) for g in dataset.graphs])
    # for g in dataset.graphs:
    #     g.create_formats_()
        
    np.random.seed(worker_info.seed % (2 ** 32))

def merge_nlist(ls:list):
    res = []
    for l in ls:
        res.extend(l)
    return res

metapath_dict = {
    'post' : ['hase', 'ein'], 
    'word' : ['win', 'hasw'], 
    'emoji' : ['by', 'withe']
}

'''
    Graph CL dataset
'''
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
        etype=('emoji', 'ein', 'post'),
        metapath=['ein', 'hase']
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
        self.length = sum([g.num_nodes(etype[-1]) for g in self.graphs])
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
        degrees = torch.cat([g.in_degrees(etype=self.etype).double() ** 0.75 for g in self.graphs])
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
                        (self.graphs[graph_idx].in_degrees(etype=self.etype)[node_idx] ** 0.75)
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
        etype=('emoji','ein','post'),
        metapath=['ein', 'hase']
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
                    (self.graphs[graph_idx].in_degrees(etype=self.etype)[node_idx] ** 0.75)
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
        
        if self.positional_embedding_size > 0:
            graph_q = data_util._add_undirected_graph_positional_embedding(graph_q, self.positional_embedding_size)
            graph_k = data_util._add_undirected_graph_positional_embedding(graph_k, self.positional_embedding_size)


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
        etype=('emoji','ein','post'),
        metapath=['ein', 'hase']
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
        edge_fanout = {}
        for et in self.graphs[0].etypes:
            edge_fanout[et] = 0
        self.fan_outs=[]
        for edge in metapath:
            fan_out = edge_fanout.copy()
            fan_out[edge]=num_neighbors
            self.fan_outs.append(fan_out)
        self.sampler = dgl.dataloading.NeighborSampler(self.fan_outs, edge_dir='in')

    def __set_seed__(self, subg, ntype):
        ntypes = subg.ntypes
        nnodes = [subg.num_nodes(n) for n in ntypes]
        seed={}
        for i,k in enumerate(ntypes):
            seed[k] = torch.zeros(nnodes[i], dtype=torch.long).to(subg.device)
        subg.ndata.update({'seed':seed})
        try:
            subg.ndata["seed"][ntype][0] = 1
        except:
            print("Error: Subgraph's len < 1 =:", len(subg.ndata["seed"][self.etype[-1]]), self.etype, subg)
        return subg

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        other_node_idx = node_idx
        ntype = self.etype[-1] # edge type

        graph_q = self.graphs[graph_idx].subgraph(
                    self.sampler.sample(self.graphs[graph_idx], {ntype:[node_idx]})[0]
                )
        graph_k = self.graphs[graph_idx].subgraph(
                    self.sampler.sample(self.graphs[graph_idx], {ntype:[other_node_idx]})[0]
                )
        
        graph_q = self.__set_seed__(graph_q, ntype)
        graph_k = self.__set_seed__(graph_k, ntype)

        if self.graph_transform:
            graph_q = self.graph_transform(graph_q)
            graph_k = self.graph_transform(graph_k)

        try:
            graph_q = dgl.to_homogeneous(graph_q, ['feat', 'seed'])
            graph_k = dgl.to_homogeneous(graph_k, ['feat', 'seed'])
        except:
            print("error")
            raise EnvironmentError

        if self.positional_embedding_size > 0:
            graph_q = data_util._add_undirected_graph_positional_embedding(graph_q, self.positional_embedding_size)
            graph_k = data_util._add_undirected_graph_positional_embedding(graph_k, self.positional_embedding_size)

        return graph_q, graph_k


'''
    Graph LP dataset
'''
def get_hete_linkpred_batch_loader(graph,
                                etype, 
                                batch_size=1,
                                num_layers=1,
                                device='cpu', shuffle=True, neg=1):
    g = graph
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(neg))
    
    train_eid_dict = {
        etype: g.edges(etype=etype, form='eid')
        # for etype in g.canonical_etypes
        }
    
    dataloader = dgl.dataloading.DataLoader(
        g, train_eid_dict, sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        device=device
    )
    
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
        num_samples=10000,
        positional_embedding_size=32
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
        self.positional_embedding_size = positional_embedding_size

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
                    print("Error: Subgraph's len < 1 =:", len(subg.ndata["seed"][self.etype[-1]]), self.etype, subg)
                
                subg = dgl.to_homogeneous(subg, ['feat', 'seed'])
                if self.positional_embedding_size > 0:
                    subg = data_util._add_undirected_graph_positional_embedding(subg, self.positional_embedding_size)

                subgs.append(subg)

        return  dgl.to_homogeneous(pos_graph), dgl.to_homogeneous(neg_graph), dgl.batch(subgs)

class LoadLPHeteGraphDataset2(LoadLPHeteGraphDataset):
    def __init__(
        self,
        etype=('emoji','ein','post'),
        num_workers=1,
        dgl_graphs_file="/home/tandz/emojis/unsupervised_emojis/xhs/data/graph_bin/train_hg3.bin",
        device='cpu',
        batch_size=1024,
        num_samples=10000,
        num_neighbour=5,
        metapath=['hase', 'ein'],
        positional_embedding_size=32
    ):  
        super().__init__(
            etype,
            num_workers,
            dgl_graphs_file,
            device,
            batch_size,
            num_samples,
            positional_embedding_size
        )
        edge_fanout = {}
        for et in self.graphs[0].etypes:
            edge_fanout[et] = 0
        
        self.sampler={}
        for ntype in [etype[0], etype[-1]]:
            fan_outs=[]
            for edge in metapath_dict[ntype]:
                fan_out = edge_fanout.copy()
                fan_out[edge] = num_neighbour
                fan_outs.append(fan_out)
            self.sampler[ntype] = dgl.dataloading.NeighborSampler(fan_outs, edge_dir='in')


        fan_outs=[]
        for edge in metapath_dict['word']:
            fan_out = edge_fanout.copy()
            if edge in ('hasw'):
                fan_out['by'] = num_neighbour // 2
                fan_out['hasw'] = num_neighbour // 2
            else:
                fan_out[edge] = num_neighbour
            fan_outs.append(fan_out)
        self.sampler['backup'] = dgl.dataloading.NeighborSampler(fan_outs, edge_dir='in')

    def __getitem__(self, data):
        input_nodes, pos_graph, neg_graph, mfgs = data
        subgs=[]
        # for ntype in pos_graph.ntypes:
        #     for nid in pos_graph.ndata['_ID'][ntype]:
        #         subgs.append(dgl.subgraph(self.graphs[0], self.sampler.sample(self.graphs[0], {ntype:[nid]})[0]))
        for ntype in pos_graph.ndata['_ID']:
            for nid in pos_graph.ndata['_ID'][ntype]:
                subg = self.graphs[0].subgraph(self.sampler[ntype].sample(self.graphs[0], {ntype:[nid.item()]})[0])
                if subg.num_edges() == 0:
                    subg = self.graphs[0].subgraph(self.sampler['backup'].sample(self.graphs[0], {ntype:[nid.item()]})[0])
                    
                ntypes = subg.ntypes
                nnodes = [subg.num_nodes(n) for n in ntypes]
                seed={}
                for i,k in enumerate(ntypes):
                    seed[k] = torch.zeros(nnodes[i], dtype=torch.long).to(subg.device)
                subg.ndata.update({'seed':seed})
                try:
                    subg.ndata["seed"][ntype][0] = 1
                except:
                    print("Error: Subgraph's len < 1 =:", len(subg.ndata["seed"][self.etype[-1]]), self.etype, subg)
                subg = dgl.to_homogeneous(subg, ['feat', 'seed'])
                
                if self.positional_embedding_size > 0:
                    subg = data_util._add_undirected_graph_positional_embedding(subg, self.positional_embedding_size)

                subgs.append(subg)

        return  dgl.to_homogeneous(pos_graph), dgl.to_homogeneous(neg_graph), dgl.batch(subgs)

'''
    Graph Infer dataset
'''
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
                    self.graphs[graph_idx].in_degrees(node_idx, etype=self.etype)
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
        etype=('emoji','ein','post'),
        metapath=['ein', 'hase']
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
        self.length = sum([g.num_nodes(etype[-1]) for g in self.graphs])
        self.total = self.length

class NodeClassificationDatasetv2(NodeClassificationDataset):
    def __init__(
        self,
        dgl_graphs_file="/home/tandz/emojis/unsupervised_emojis/data/graph_bin/data.bin",
        rw_hops=64,
        subgraph_size=64,
        restart_prob=0.5,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
        device='cpu',
        etype=('emoji','ein','post'),
        metapath=['ein', 'hase'],
        num_neighbor=5
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
        edge_fanout = {}
        for et in self.graphs[0].etypes:
            edge_fanout[et] = 0
        self.fan_outs=[]
        for edge in metapath:
            fan_out = edge_fanout.copy()
            fan_out[edge]=num_neighbor
            self.fan_outs.append(fan_out)
        self.sampler = dgl.dataloading.NeighborSampler(self.fan_outs, edge_dir='in')
        if etype[-1] == 'word':
            self.fan_outs[1]['hasw']=num_neighbor//2
            self.fan_outs[1]['by']=num_neighbor//2
            self.sampler_bk = dgl.dataloading.NeighborSampler(self.fan_outs, edge_dir='in')
    def __set_seed__(self, subg, ntype):
        ntypes = subg.ntypes
        nnodes = [subg.num_nodes(n) for n in ntypes]
        seed={}
        for i,k in enumerate(ntypes):
            seed[k] = torch.zeros(nnodes[i], dtype=torch.long).to(subg.device)
        subg.ndata.update({'seed':seed})
        try:
            subg.ndata["seed"][ntype][0] = 1
        except:
            print("Error: Subgraph's len < 1 =:", len(subg.ndata["seed"][self.etype[-1]]), self.etype, subg)
        return subg

    def __getitem__(self, idx):
        graph_idx, node_idx = self._convert_idx(idx)
        other_node_idx = node_idx
        ntype = self.etype[-1]
        subg = self.graphs[0].subgraph(self.sampler.sample(self.graphs[0], {ntype:[node_idx]})[0])
        if subg.num_edges() == 0:
            subg = self.graphs[0].subgraph(self.sampler_bk.sample(self.graphs[0], {ntype:[node_idx]})[0])
        subg = self.__set_seed__(subg, ntype)
        graph_q = dgl.to_homogeneous(subg, ['feat', 'seed'])
        if self.positional_embedding_size > 0:
            graph_q = data_util._add_undirected_graph_positional_embedding(graph_q, self.positional_embedding_size)
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


if __name__ == "__main__":
    num_workers = 1
    import psutil

    mem = psutil.virtual_memory()
    print(mem.used / 1024 ** 3)
    g=dgl.data.utils.load_graphs("data/graph_bin/small_test.bin")[0][0]
    edge_fanout = {}
    for et in g.etypes:
        edge_fanout[et] = 0
    edge_fanout['withe'] = -1
    graph_q = dgl.sampling.sample_neighbors(
        g,
        {'emoji':0},
        edge_fanout,
        edge_dir='in'
    )



    edge_fanout = {
        'by': 0, 
        'ein': 0, 
        'hase': 0, 
        'hasw': 0, 
        'win': 0, 
        'withe': 0
    }
    fan_outs=[]
    for edge in ['ein']:
        fan_out = edge_fanout.copy()
        fan_out[edge]=5
        fan_outs.append(fan_out)
    sampler = dgl.dataloading.NeighborSampler(fan_outs)
    subg = g.subgraph(sampler.sample(g, {'post':[1]})[0])

    exit(0)

    # #remove edges from graph
    # eids=g.edges('all', etype='win')[-1]
    # g.remove_edges(eids=eids, etype='win')
    # # get all etypes
    # g.etypes ---> ['by', 'ein', 'hase', 'hasw', 'win', 'withe']
