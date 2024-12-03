#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import time

import dgl
import numpy as np
import torch

from gcc.datasets.graph_dataset_new import (
    LoadLPHeteGraphDataset2,
    GraphCLDataset_NS,
    NodeClassificationDatasetv2
)

from gcc.datasets.data_util import batcher
from gcc.models import GraphEncoder
import tqdm

import logging

# from gen_emojis import gen_emoji
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
logger = logging.getLogger('Generate')


def test_moco(train_loader, model, opt, etype):
    """
    one epoch training for moco
    """
    model.eval()
    emb_list = []
    for idx, batch in enumerate(tqdm.tqdm(train_loader)):
        graph_q, graph_k = batch
        bsz = graph_q.batch_size
        graph_q = graph_q.to(opt.device)
        graph_k = graph_k.to(opt.device)
        # print(graph_q.ndata['oridx'])
        with torch.no_grad():
            feat_q = model(graph_q, etype)
            feat_k = model(graph_k, etype)

        assert feat_q.shape == (bsz, opt.hidden_size)
        emb_list.append(((feat_q + feat_k) / 2).detach().cpu())
    return torch.cat(emb_list)

def main(args_test, dgl_graphs_file):
    if os.path.isfile(args_test.load_path):
        logger.info("=> loading checkpoint '{}'".format(args_test.load_path))
        checkpoint = torch.load(args_test.load_path, map_location="cpu")
        logger.info(
            "=> loaded successfully '{}' (epoch {})".format(
                args_test.load_path, checkpoint["epoch"]
            )
        )
        epoch=checkpoint["epoch"]
    else:
        logger.warn("=> no checkpoint found at '{}'".format(args_test.load_path))
        return
    args = checkpoint["opt"]
    print(args)
    assert args_test.gpu is None or torch.cuda.is_available()
    logger.info("Use GPU: {} for generation".format(args_test.gpu))
    args.gpu = args_test.gpu
    args.device = torch.device("cpu") if args.gpu is None else torch.device(args.gpu)

    train_dataset = NodeClassificationDatasetv2(
        dgl_graphs_file=dgl_graphs_file,
        rw_hops=args.rw_hops,
        subgraph_size=args.subgraph_size,
        restart_prob=args.restart_prob,
        positional_embedding_size=args.positional_embedding_size,
        device=torch.device(args.gpu),
        etype=args_test.etype,
        metapath=args_test.metapath
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=batcher(),
        shuffle=False,
    )

    model = GraphEncoder(
            positional_embedding_size=args.positional_embedding_size,
            max_degree=args.max_degree,
            degree_embedding_size=args.degree_embedding_size,
            output_dim=args.hidden_size,
            node_hidden_dim=args.hidden_size,
            node_feat_dim=args.node_feat_dim,
            num_layers=args.num_layer,
            norm=args.norm,
            degree_input=True,
        )
    model = model.to(args.device)

    model.load_state_dict(checkpoint["model"])
    del checkpoint
    emb = test_moco(train_loader, model, args, args_test.etype)
    torch.save(emb.detach(), f"output/{args_test.etype[-1]}.pth")
    logger.info(f"pred done! save path: output/{args_test.etype[-1]}.pth")
    return emb[0].detach()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for Inference")
    # fmt: offm
    parser.add_argument("--load-path", type=str, help="path to load model")
    parser.add_argument("--gpu", default=1, type=int, help="GPU id to use.")
    parser.add_argument("--ntype", default="2", type=str, help="node type, 0:post, 1:word, 2:emoji")
    parser.add_argument("--dgl_file", type=str, default="data/graph_bin/small_test.bin", help="dgl grapn file path")
    # fmt: on
    args_test=parser.parse_args()
    dgl_graphs_file = args_test.dgl_file

    etypes = [('emoji', 'ein', 'post'), ('post', 'hasw', 'word'), ('word', 'withe', 'emoji')]
    metapaths=[['hase', 'ein'], ['win', 'hasw'], ['by', 'withe']]

    res = []
    for n in args_test.ntype.split(","):
        n = int(n)
        args_test.etype = etypes[n]
        args_test.metapath = metapaths[n]
        print(args_test.etype, args_test.metapath)
        emb = main(args_test, dgl_graphs_file)
        res.append(emb)


