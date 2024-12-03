#!/usr/bin/env python
# encoding: utf-8
# File Name: graph_encoder.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/31 18:42
# TODO:

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import Set2Set

from gcc.models.gin import NodeGIN, RGCN

from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


class ScorePredictor(nn.Module):
    def __init__(self):
        super(ScorePredictor, self).__init__()

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            edge_subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'))
            return edge_subgraph.edata['score']


class GraphMergeLayer(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        final_dropout,
        graph_pooling_type="sum",
        norm=False,
    ):
        super(GraphMergeLayer, self).__init__()
        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == "sum":
            self.pool = SumPooling()
        elif graph_pooling_type == "mean":
            self.pool = AvgPooling()
        elif graph_pooling_type == "max":
            self.pool = MaxPooling()
        else:
            raise NotImplementedError
        
        self.norm = norm
        
    def forward(self, g, hidden_rep:list, return_all_outputs=False):
        """
        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph for molecule(s)

        hidden_rep: all conv layers outputs include ori feat
        """
        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        all_outputs = []
        for i, h in list(enumerate(hidden_rep)):
            pooled_h = self.pool(g, h)
            all_outputs.append(pooled_h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        x = score_over_layer
        if self.norm:
            x = F.normalize(x, p=2, dim=-1, eps=1e-5)

        if return_all_outputs:
            return x, all_outputs
        else:
            return x


class Model(nn.Module):
    def __init__(
        self,
        positional_embedding_size=32,
        max_degree=128,
        degree_embedding_size=32,
        output_dim=32,
        node_hidden_dim=32,
        node_feat_dim=768,
        num_layers=2,
        norm=False,
        degree_input=False,
        device='cpu'):

        super(Model, self).__init__()
        
        self.degree_input = degree_input
        self.max_degree = max_degree
        self.device = device
        # build input
        if degree_input:
            node_input_dim = node_feat_dim + degree_embedding_size + 1 + positional_embedding_size
            self.degree_embedding = nn.Embedding(
                num_embeddings=max_degree + 1, embedding_dim=degree_embedding_size
            )
        else:
            node_input_dim = node_feat_dim + 1 + positional_embedding_size


        # backbone
        self.gnn = NodeGIN(
            num_layers=num_layers-1,
            num_mlp_layers=2,
            input_dim=node_input_dim,
            hidden_dim=node_hidden_dim,
            learn_eps=False,
            neighbor_pooling_type="sum",
            use_selayer=False,
        )


        # output task
        self.cl_task = GraphMergeLayer(
                        num_layers,
                        node_input_dim,
                        node_hidden_dim,
                        output_dim,
                        final_dropout=0.5,
                        graph_pooling_type="sum",
                        norm=norm,)

        self.pl_task = ScorePredictor()

    
    def build_input(self, g):
        if isinstance(g, list):
            g=g[0]
        
        try:
            if isinstance(g.ndata[''], dict):
                emb=g.ndata["feat"]['_N']
                seed = torch.zeros(emb.shape[0], dtype=torch.long).to(emb.device)
                pos_undirected=g.ndata["pos_undirected"]['_N']
        except:
            pos_undirected=g.ndata["pos_undirected"]
            emb=g.ndata["feat"]
            try:
                seed=g.ndata["seed"]
            except:
                seed = torch.zeros(emb.shape[0], dtype=torch.long).to(emb.device)


        if self.degree_input:
            device = seed.device
            # try:
            #     degrees = g.in_degrees()
            # except:
            degrees = g.out_degrees()
                
            if device != torch.device("cpu"):
                degrees = degrees.cuda(device)
            # print(emb.shape, seed.shape, degrees.shape)
            n_feat = torch.cat(
                (
                    pos_undirected,
                    self.degree_embedding(degrees.clamp(0, self.max_degree)),
                    seed.unsqueeze(1).float(),
                    emb
                ),
                dim=-1,
            )
        else:
            n_feat = torch.cat(
                (
                    pos_undirected,
                    seed.unsqueeze(1).float(),
                    emb
                ),
                dim=-1,
            )
        return n_feat


    def forward(self, g, edtype, return_all_outputs=False):
        """
        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph for molecule(s)
        n_feat : tensor of dtype float32 and shape (B1, D1)
            Node features. B1 for number of nodes and D1 for
            the node feature size.
        e_feat : tensor of dtype float32 and shape (B2, D2)
            Edge features. B2 for number of edges and D2 for
            the edge feature size.

        Returns
        -------
        res : Predicted labels
        """

        # build input
        n_feat = self.build_input(g)

        # backbone h_reps is a list contains all convlayers' output
        h_reps = self.gnn(g, n_feat)

        # outout tasks
        x = self.cl_task(g, h_reps, return_all_outputs=return_all_outputs)

        return x


    def encode(self, g, edtype):
        """
            retrun gnn encoder's last layer's output
        """
        # build input
        n_feat = self.build_input(g)

        # backbone h_reps is a list contains all convlayers' output
        h_reps = self.gnn(g, n_feat)

        return h_reps


class HetegModel(nn.Module):
    def __init__(
        self,
        positional_embedding_size=32,
        max_degree=128,
        degree_embedding_size=32,
        output_dim=32,
        node_hidden_dim=32,
        node_feat_dim=768,
        num_layers=2,
        norm=False,
        degree_input=False,
        device='cpu'):

        super(HetegModel, self).__init__()
        
        self.degree_input = degree_input
        self.max_degree = max_degree
        self.device = device
        # build input
        if degree_input:
            node_input_dim = node_feat_dim + degree_embedding_size + 1 # + positional_embedding_size
            self.degree_embedding = nn.Embedding(
                num_embeddings=max_degree + 1, embedding_dim=degree_embedding_size
            )
        else:
            node_input_dim = node_feat_dim + 1 # + positional_embedding_size


        # backbone
        self.gnn = RGCN(
            in_feats=node_input_dim,
            hid_feats=node_hidden_dim,
            num_layers=num_layers-1,
            rel_names=['post','emoji']
        )


        # output task
        self.cl_task = GraphMergeLayer(
                        num_layers,
                        node_input_dim,
                        node_hidden_dim,
                        output_dim,
                        final_dropout=0.5,
                        graph_pooling_type="sum",
                        norm=norm,)

        self.pl_task = ScorePredictor()

    
    def build_input(self, g, etype):
        if isinstance(g, list):
            g=g[0]
        
        # try:
        #     if isinstance(g.ndata[''], dict):
        #         emb=g.ndata["emb"]['_N']
        #         seed = torch.zeros(emb.shape[0], dtype=torch.long).to(emb.device)
        #         # pos_undirected=g.ndata["pos_undirected"]['_N']
        # except:
        #     # pos_undirected=g.ndata["pos_undirected"]
        #     emb=g.ndata["emb"]
        #     try:
        #         seed=g.ndata["seed"]
        #     except:
        #         seed = torch.zeros(emb.shape[0], dtype=torch.long).to(emb.device)
        ntypes = g.ntypes
        nnodes = [g.num_nodes(n) for n in ntypes]

        emb = g.ndata['feat']
        # try:
        #     seed=g.ndata["seed"]
        # except:
        seed={}
        for i,k in enumerate(ntypes):
            seed[k] = g.ndata["seed"].get(k, torch.zeros(nnodes[i], dtype=torch.long).to(self.device))
        g.ndata.update({'seed':seed})

        if self.degree_input:
            # try:
            #     degrees = g.in_degrees()
            # except:
            degrees = {}
            inde = g.in_degrees(etype=etype)
            outde = g.out_degrees(etype=etype)
            degrees[etype[0]] = outde.cuda(self.device)
            degrees[etype[-1]] = inde.cuda(self.device)

            # if self.device != torch.device("cpu"):
            #     degrees = degrees.cuda(self.device)
            # print(emb.shape, seed.shape, degrees.shape)
            n_feat={}
            for i,k in enumerate(ntypes):
                n_feat[k] = torch.cat(
                    (
                        # pos_undirected,
                        self.degree_embedding(degrees.get(k,torch.zeros(nnodes[i], device=self.device, dtype=torch.long)).clamp(0, self.max_degree)),
                        seed[k].unsqueeze(1).float(),
                        emb[k]
                    ),
                    dim=-1,
                )
        else:
            n_feat={}
            for i,k in enumerate(ntypes):
                n_feat[k] = torch.cat(
                (
                    # pos_undirected,
                    seed[k].unsqueeze(1).float(),
                    emb[k]
                ),
                dim=-1,
            )
            
        return n_feat


    def forward(self, g, etype, return_all_outputs=False):
        """
        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph for molecule(s)
        n_feat : tensor of dtype float32 and shape (B1, D1)
            Node features. B1 for number of nodes and D1 for
            the node feature size.
        e_feat : tensor of dtype float32 and shape (B2, D2)
            Edge features. B2 for number of edges and D2 for
            the edge feature size.

        Returns
        -------
        res : Predicted labels
        """

        # build input
        n_feat = self.build_input(g, etype)
        
        # backbone h_reps is a list contains all convlayers' output
        # 多层gnn
        h_reps = self.gnn(g, n_feat)

        # outout tasks
        # g.ndata.update({'n_feat':n_feat})
        # g = dgl.to_homogeneous(g, ndata=g.ndata.keys())
        # n_feat = g.ndata['n_feat']
        x = self.cl_task(g, h_reps, return_all_outputs=return_all_outputs)

        return x


    def encode(self, g, etype):
        """
            retrun gnn encoder's last layer's output
        """
        # build input
        n_feat = self.build_input(g, etype)

        # backbone h_reps is a list contains all convlayers' output
        h_reps = self.gnn(g, n_feat)

        return h_reps[-1]





if __name__ == "__main__":
    model = Model(positional_embedding_size=16, node_feat_dim=16)
    print(model)
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 0, 1, 2], [1, 2, 2, 1])
    g.ndata["pos_directed"] = torch.rand(3, 16)
    g.ndata["pos_undirected"] = torch.rand(3, 16)
    g.ndata["emb"] = torch.rand(3, 16)
    g.ndata["seed"] = torch.zeros(3, dtype=torch.long)
    g.ndata["nfreq"] = torch.ones(3, dtype=torch.long)
    g.edata["efreq"] = torch.ones(4, dtype=torch.long)
    g = dgl.batch([g, g, g])
    y = model(g)
    print(y.shape)
    print(y)

    y = model.encode(g)
    print(y.shape)
    print(y)
