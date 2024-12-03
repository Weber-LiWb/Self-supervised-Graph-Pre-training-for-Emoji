
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
import torch_geometric
import pandas as pd
import scipy.sparse as sp

def _create_dgl_graph(data):
    graph = dgl.DGLGraph()
    src, dst = data.edge_index.tolist()
    num_nodes = data.edge_index.max() + 1
    graph.add_nodes(num_nodes)
    graph.add_edges(src, dst)
    graph.add_edges(dst, src)
    graph.readonly()
    return graph

if __name__ == "__main__":
    pass