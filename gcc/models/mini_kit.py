#%% 
import re
import numpy as np
import pandas as pd
import torch
import random
from xhs.gen_bert_code import encode_func
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import Module, init
from torch import Tensor
import math
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
#%%

class ScoreLpPredicter(Module):
    def __init__(self, in_feat, out_feat=500, activation='sigmoid'):
        super().__init__()
        self.k1 = torch.nn.Linear(in_feat, in_feat)
        self.k2 = torch.nn.Linear(in_feat, in_feat)
        self.l1 = torch.nn.Linear(2*in_feat, in_feat)
        self.l2 = torch.nn.Linear(in_feat, out_feat)
        # self.act = torch.nn.Sigmoid() if activation == 'sigmoid' else torch.nn.ReLU()
    
    def forward(self, post, emoji):
        pshape = post.shape
        eshape = emoji.shape
        post  = self.k1(post)
        emoji = self.k2(emoji)
        post  = torch.tile(post.unsqueeze(1),  [1, eshape[0], 1])
        emoji = torch.tile(emoji.unsqueeze(0), [pshape[0], 1, 1])
        x = torch.cat([post, emoji], dim=-1)
        x = F.relu(self.l2(F.relu(self.l1(x))))
        return x

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
        h = torch.cat([self.k1(edges.src["h"]), self.k2(edges.dst["h"])], 1)
        return {"score": F.relu(self.l2(F.relu(self.l1(h))))}

    def graph_forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


class ScoreLpPredicter2(Module):
    def __init__(self, in_feat, out_feat=1, activation='sigmoid'):
        super().__init__()
        self.k1 = torch.nn.Linear(in_feat, in_feat)
        self.k2 = torch.nn.Linear(in_feat, in_feat)
        self.l1 = torch.nn.Linear(2*in_feat, in_feat)
        self.l2 = torch.nn.Linear(in_feat, out_feat)
        # self.act = torch.nn.Sigmoid() if activation == 'sigmoid' else torch.nn.ReLU()
    
    def forward(self, post, emoji):
        pshape = post.shape
        eshape = emoji.shape
        post  = self.k1(post)
        emoji = self.k2(emoji)
        post  = torch.tile(post.unsqueeze(1),  [1, eshape[0], 1])
        emoji = torch.tile(emoji.unsqueeze(0), [pshape[0], 1, 1])
        x = torch.cat([post, emoji], dim=-1)
        x = F.sigmoid(self.l2(F.relu(self.l1(x))))
        return x.squeeze(-1)

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
        h = torch.cat([self.k1(edges.src["h"]), self.k2(edges.dst["h"])], 1)
        # h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        return {"score": F.sigmoid(self.l2(F.relu(self.l1(h.squeeze(1)))))}

    def graph_forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


class ScoreDotPredicter(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, post, emoji):
        return torch.mm(post, emoji.T)