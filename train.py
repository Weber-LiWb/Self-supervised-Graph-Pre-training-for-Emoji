#!/usr/bin/env python
# encoding: utf-8

import argparse
import copy
import os
import time
import warnings
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
logger = logging.getLogger('Train')
import dgl  # package for sampling
import numpy as np
import psutil
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
from gcc.contrastive.criterions import NCESoftmaxLoss, NCESoftmaxLossNS
from gcc.contrastive.memory_moco import MemoryMoCo
from gcc.datasets.graph_dataset_new import (
    LoadLPHeteGraphDataset2,
    GraphCLDataset_NS
)
from gcc.datasets import (
    GRAPH_CLASSIFICATION_DSETS
)

from gcc.datasets.data_util import batcher, labeled_batcher, batcher_neg, batcher_pl
from gcc.models import GraphEncoder as HeteGraphEncoder
# from gcc.models import  NodeEncoder #, HeteGraphEncoder
from gcc.utils.misc import AverageMeter, adjust_learning_rate, warmup_linear
from sklearn.metrics import roc_auc_score

def parse_option():

    # fmt: off
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--node-types", type=int, default=2, help="node type, 0 or 1 or 2")

    parser.add_argument("--print-freq", type=int, default=10, help="print frequency")
    parser.add_argument("--tb-freq", type=int, default=250, help="tb frequency")
    parser.add_argument("--save-freq", type=int, default=1, help="save frequency")
    parser.add_argument("--batch-size", type=int, default=256, help="batch_size for moco task")
    parser.add_argument("--batch-size2", type=int, default=64, help="batch_size for lp task")
    parser.add_argument("--num-workers", type=int, default=2, help="num of workers to use")
    parser.add_argument("--num-copies", type=int, default=1, help="num of dataset copies that fit in memory")
    parser.add_argument("--num-samples", type=int, default=5000, help="num of samples per batch per worker")
    parser.add_argument("--epochs", type=int, default=15, help="number of training epochs")

    # optimization
    parser.add_argument("--optimizer", type=str, default='adam', choices=['sgd', 'adam', 'adagrad'], help="optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="learning rate")
    parser.add_argument("--lr_decay_epochs", type=str, default="120,160,200", help="where to decay lr, can be a list")
    parser.add_argument("--lr_decay_rate", type=float, default=0.0, help="decay rate for learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for Adam")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="clip norm")

    # resume
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")

    # augmentation setting
    parser.add_argument("--aug", type=str, default="1st", choices=["1st", "2nd", "all"])

    parser.add_argument("--exp", type=str, default="")

    # dataset definition
    parser.add_argument("--dataset", type=str, default="dgl", choices=["dgl", "wikipedia", "blogcatalog", "usa_airport", "brazil_airport", "europe_airport", "cora", "citeseer", "pubmed", "kdd", "icdm", "sigir", "cikm", "sigmod", "icde", "h-index-rand-1", "h-index-top-1", "h-index"] + GRAPH_CLASSIFICATION_DSETS)

    # model definition
    parser.add_argument("--model", type=str, default="gin", choices=["gat", "mpnn", "gin"])
    # other possible choices: ggnn, mpnn, graphsage ...
    parser.add_argument("--num-layer", type=int, default=2, help="gnn layers")
    parser.add_argument("--readout", type=str, default="avg", choices=["avg", "set2set"])
    parser.add_argument("--set2set-lstm-layer", type=int, default=3, help="lstm layers for s2s")
    parser.add_argument("--set2set-iter", type=int, default=6, help="s2s iteration")
    parser.add_argument("--norm", action="store_true", default=True, help="apply 2-norm on output feats")

    # loss function
    parser.add_argument("--nce-k", type=int, default=32, help="queue size")
    parser.add_argument("--nce-t", type=float, default=0.07, help="moco temperature")

    # random walk
    parser.add_argument("--rw-hops", type=int, default=4)
    parser.add_argument("--subgraph-size", type=int, default=64)
    parser.add_argument("--restart-prob", type=float, default=0.2)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--node-feat-dim", type=int, default=768)
    parser.add_argument("--positional-embedding-size", type=int, default=32)
    parser.add_argument("--max-node-freq", type=int, default=16)
    parser.add_argument("--max-edge-freq", type=int, default=16)
    parser.add_argument("--max-degree", type=int, default=512)
    parser.add_argument("--freq-embedding-size", type=int, default=16)
    parser.add_argument("--degree-embedding-size", type=int, default=16)

    # specify folder
    parser.add_argument("--output-path", type=str, default="output", help="path to save model")
    parser.add_argument("--load-path", type=str, default=None, help="loading checkpoint at test time")

    # finetune setting
    parser.add_argument("--finetune", action="store_true")

    parser.add_argument("--alpha", type=float, default=0.999, help="exponential moving average weight")
    parser.add_argument("--dgl_file", type=str, default="data/graph_bin/small_test.bin", help="dgl grapn file path")

    # GPU setting
    # parser.add_argument("--gpu", default=[], type=int, nargs='+', help="GPU id to use.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")

    # cross validation
    parser.add_argument("--seed", type=int, default=0, help="random seed.")
    # fmt: on
    # cross validation
    # memory setting
    parser.add_argument("--moco", action="store_true", help="Graph Cl Task")
    parser.add_argument("--linkpred", action="store_true", help="Graph linkpred task")
    parser.add_argument("--moco_type", type=str, default='0,1,2', help="moco etypes")
    parser.add_argument("--lp_type", type=str, default='0,1,2', help="lp etypes")

    parser.add_argument("--version", type=str, default="", help="model version")

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt

def option_update(opt):
    opt.model_name = "moco_{}_linkpred_{}".format(
        opt.moco,
        opt.linkpred,
    )

    if opt.load_path is None:
        opt.model_folder = os.path.join(opt.model_path, opt.model_name)
        if not os.path.isdir(opt.model_folder):
            os.makedirs(opt.model_folder)
    else:
        opt.model_folder = opt.load_path

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    return opt

def get_optimizer(args, model):
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=args.learning_rate,
            lr_decay=args.lr_decay_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError

    return optimizer

def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    return roc_auc_score(labels, scores)

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).squeeze()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0], device=scores.device), torch.zeros(neg_score.shape[0], device=scores.device)]
    )
    return F.binary_cross_entropy_with_logits(scores, labels)

def clip_grad_norm(params, max_norm):
    """Clips gradient norm."""
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(
            sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None)
        )



def train_moco(
    epoch, train_loader, model, model_ema, contrast, criterion, optimizer, sw, opt, etype
):
    """
    one epoch training for moco
    """
    n_batch = train_loader.dataset.total // opt.batch_size
    model.train()
    model_ema.eval()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.train()

    model_ema.apply(set_bn_train)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    epoch_loss_meter = AverageMeter()
    prob_meter = AverageMeter()
    graph_size = AverageMeter()
    gnorm_meter = AverageMeter()
    max_num_nodes = 0
    max_num_edges = 0

    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        graph_q, graph_k = batch
        # graph_q = graph_q.to(torch.device(opt.gpu))
        # graph_k = graph_k.to(torch.device(opt.gpu))
        bsz = graph_q.batch_size
        # print(graph_k.device, graph_q.device)
        # if opt.moco:
        # ===================Moco forward=====================
        feat_q = model(graph_q, etype)
        with torch.no_grad():
            feat_k = model_ema(graph_k, etype)
        out = contrast(feat_q, feat_k)
        prob = out[:, 0].mean()

        assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)

        # ===================backward=====================
        optimizer.zero_grad()
        loss = criterion(out)
        loss.backward()
        grad_norm = clip_grad_norm(model.parameters(), opt.clip_norm)

        global_step = epoch * n_batch + idx
        lr_this_step = opt.learning_rate * warmup_linear(
            global_step / (opt.epochs * n_batch), 0.1
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step
        optimizer.step()

        # ===================meters=====================
        # print("loss:",loss.item())
        loss_meter.update(loss.item(), bsz)
        epoch_loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)
        graph_size.update(
            (graph_q.number_of_nodes() + graph_k.number_of_nodes()) / 2.0 / bsz, 2 * bsz
        )
        gnorm_meter.update(grad_norm, 1)
        max_num_nodes = max(max_num_nodes, graph_q.number_of_nodes())
        max_num_edges = max(max_num_edges, graph_q.number_of_edges())

        if opt.moco:
            moment_update(model, model_ema, opt.alpha)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            mem = psutil.virtual_memory()
            #  print(f'{idx:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')
            #  mem_used.append(mem.used/1024**3)
            logger.info(
                "Train: [{0}][{1}/{2}]\t"
                # "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                # "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.5f} ({loss.avg:.5f})\t"
                "prob {prob.val:.3f} ({prob.avg:.3f})\t"
                "GS {graph_size.val:.3f} ({graph_size.avg:.3f})\t"
                "LR {LR:.5f}".format(
                    epoch,
                    idx + 1,
                    n_batch,
                    # batch_time=batch_time,
                    # data_time=data_time,
                    loss=loss_meter,
                    prob=prob_meter,
                    graph_size=graph_size,
                    LR=lr_this_step#mem=mem.used / 1024 ** 3,
                )
            )
            #  print(out[0].abs().max())

        # tensorboard logger
        if (idx + 1) % opt.tb_freq == 0:
            global_step = epoch * n_batch + idx
            sw.add_scalar("moco_loss", loss_meter.avg, global_step)
            sw.add_scalar("moco_prob", prob_meter.avg, global_step)
            sw.add_scalar("graph_size", graph_size.avg, global_step)
            sw.add_scalar("graph_size/max", max_num_nodes, global_step)
            sw.add_scalar("graph_size/max_edges", max_num_edges, global_step)
            sw.add_scalar("gnorm", gnorm_meter.avg, global_step)
            sw.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)
            loss_meter.reset()
            prob_meter.reset()
            graph_size.reset()
            gnorm_meter.reset()
            max_num_nodes, max_num_edges = 0, 0
    return epoch_loss_meter.avg


def train_lp2(epoch, train_loader, model, criterion, optimizer, sw, opt, etype):
    """
    one epoch training for moco
    """
    bsz = train_loader.dataset.batch_size
    n_batch = train_loader.dataset.__len__()//train_loader.dataset.batch_size
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    epoch_loss_meter = AverageMeter()
    prob_meter = AverageMeter()
    graph_size = AverageMeter()
    pos_edge_size = AverageMeter()
    neg_edge_size = AverageMeter()
    gnorm_meter = AverageMeter()
    max_num_nodes = 0
    max_num_edges = 0

    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        pos_graph, neg_graph, subgs = batch
        pos_g = pos_graph#.to(torch.device(opt.gpu))
        neg_g = neg_graph#.to(torch.device(opt.gpu))
        subgs = subgs#.to(torch.device(opt.gpu))

        # ===================forward=====================
        h = model(subgs, etype)
        pos_score, neg_score = model.pl_task(pos_g, h), model.pl_task(neg_g, h)
        # ===================backward=====================
        loss = criterion(pos_score, neg_score)
        # print("idx", idx, "loss:", loss)
        loss_meter.update(loss.item())
        graph_size.update(
            (subgs.number_of_nodes() + subgs.number_of_nodes()) / 2.0 / bsz, 2 * bsz
        )
        pos_edge_size.update(
            pos_g.num_edges()
        )
        neg_edge_size.update(
            neg_g.num_edges()
        )
        epoch_loss_meter.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print info
        if (idx + 1) % opt.print_freq == 0:
            mem = psutil.virtual_memory()
            #  print(f'{idx:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')
            #  mem_used.append(mem.used/1024**3)
            logger.info(
                "Train: [{0}][{1}/{2}]\t"
                "loss {loss.val:.5f} ({loss.avg:.5f})\t"
                "PES {pos_edge_size.val:.3f} ({pos_edge_size.avg:.3f})\t"
                "NES {neg_edge_size.val:.3f} ({neg_edge_size.avg:.3f})\t"
                "GS {graph_size.val:.3f} ({graph_size.avg:.3f})\t".format(
                    epoch,
                    idx + 1,
                    n_batch,
                    loss=loss_meter,
                    pos_edge_size=pos_edge_size,
                    neg_edge_size=neg_edge_size,
                    graph_size=graph_size,
                )
            )
            #  print(out[0].abs().max())

        # tensorboard logger
        if (idx + 1) % opt.tb_freq == 0:
            global_step = epoch * n_batch + idx
            sw.add_scalar("moco_loss", loss_meter.avg, global_step)
            sw.add_scalar("moco_prob", prob_meter.avg, global_step)
            sw.add_scalar("graph_size", graph_size.avg, global_step)
            sw.add_scalar("graph_size/max", max_num_nodes, global_step)
            sw.add_scalar("graph_size/max_edges", max_num_edges, global_step)
            sw.add_scalar("gnorm", gnorm_meter.avg, global_step)
            sw.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)
            loss_meter.reset()
            prob_meter.reset()
            graph_size.reset()
            gnorm_meter.reset()
            max_num_nodes, max_num_edges = 0, 0
    return loss_meter.avg


def main(args):
    # env prepare
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args = option_update(args)
    print(args)
    assert args.gpu is not None and torch.cuda.is_available()
    logger.info("Use GPU: {} for training".format(args.gpu))
    assert args.positional_embedding_size % 2 == 0
    logger.info("setting random seeds")

    ##################################
    # dataset construct 
    ##################################
    mem = psutil.virtual_memory()
    print("before construct dataset", mem.used / 1024 ** 3)
    if args.moco:
        train_moco_dataloader = []
        for etype, metapath in zip(args.etypes, args.metapaths):
            train_dataset = GraphCLDataset_NS(
                rw_hops=args.rw_hops,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
                num_workers=args.num_workers,
                num_samples=args.num_samples,
                dgl_graphs_file=args.dgl_file,
                num_copies=args.num_copies,
                node_types=args.node_types,
                device=torch.device(args.gpu),
                etype=etype,
                metapath=metapath,
                num_neighbors=5
            )
            mem = psutil.virtual_memory()
            print("before construct dataloader", mem.used / 1024 ** 3)
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=args.batch_size,
                collate_fn=labeled_batcher() if args.finetune else batcher(),
                shuffle=True if args.finetune else False,
            )
            train_moco_dataloader.append(train_loader)
    if args.linkpred:
        train_lp_dataloader = []
        for etype, metapath in zip(args.lp_etypes, args.lp_metapaths):
            train_dataset = LoadLPHeteGraphDataset2(
                dgl_graphs_file=args.dgl_file,
                etype=etype,
                batch_size=args.batch_size2,
                device=torch.device(args.gpu),
                num_samples=args.num_samples/2,
                num_neighbour=5,
                metapath=metapath
            )
            mem = psutil.virtual_memory()
            print("before construct dataloader", mem.used / 1024 ** 3)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=1,
                collate_fn=batcher_pl(),
            )
            train_lp_dataloader.append(train_loader)
        
    ##################################
    # model construct
    ##################################
    mem = psutil.virtual_memory()
    print("before training", mem.used / 1024 ** 3)
    # create model and optimizer
    # n_data = train_dataset.total
    n_data = None
    model = HeteGraphEncoder(
            positional_embedding_size=args.positional_embedding_size,
            max_degree=args.max_degree,
            degree_embedding_size=args.degree_embedding_size,
            output_dim=args.hidden_size,
            node_hidden_dim=args.hidden_size,
            node_feat_dim=args.node_feat_dim,
            num_layers=args.num_layer,
            norm=args.norm,
            degree_input=True,
            device=torch.device(args.gpu)
        )
    model = model.cuda(args.gpu)
    if args.moco:
        model_ema = HeteGraphEncoder(
                positional_embedding_size=args.positional_embedding_size,
                max_degree=args.max_degree,
                degree_embedding_size=args.degree_embedding_size,
                output_dim=args.hidden_size,
                node_hidden_dim=args.hidden_size,
                node_feat_dim=args.node_feat_dim,
                num_layers=args.num_layer,
                norm=args.norm,
                degree_input=True,
                device=torch.device(args.gpu)
            )

        # copy weights from `model' to `model_ema'
        # moment_update(model, model_ema, 0)

        # set the contrast memory and criterion
        contrast = [MemoryMoCo(
            args.hidden_size, n_data, args.nce_k, args.nce_t, use_softmax=True
        ).cuda(args.gpu) for i in range(len(args.etypes))]
        # if len(contrast)==1:
        #     contrast=contrast[0]

        criterion = NCESoftmaxLoss()
        criterion = criterion.cuda(args.gpu)
        model_ema = model_ema.cuda(args.gpu)
    print(model)
    ##################################
    # optimaizer init
    ##################################
    optimizer = get_optimizer(args, model)

    ##################################
    # optionally resume from a checkpoint
    ##################################
    args.start_epoch = 0
    if args.resume:
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location="cpu")

        # args.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])

        # optimizer.load_state_dict(checkpoint["optimizer"])
        try:
            contrast.load_state_dict(checkpoint["contrast"])
        except:
            for i in range(len(contrast)):
                contrast[i].load_state_dict(checkpoint["contrast"][i])
        if args.moco:
            model_ema.load_state_dict(checkpoint["model_ema"])

        logger.info(
            "=> loaded successfully '{}' (epoch {})".format(
                args.resume, checkpoint["epoch"]
            )
        )
        args.start_epoch = checkpoint["epoch"]
        del checkpoint
        torch.cuda.empty_cache()
    
    ##################################
    # tensorboard
    ##################################
    sw = SummaryWriter(args.tb_folder)

    ##################################
    # train routine
    ##################################
    for epoch in range(args.start_epoch, args.epochs + 1):
        adjust_learning_rate(epoch, args, optimizer)
        logger.info("==> training...")
        time1 = time.time()
        if args.moco:
            kk=1
            if args.linkpred: 
                logger.info("CONTRASIVE TURN:")
                kk=10
            # copy weights from `model' to `model_ema'
            moment_update(model, model_ema, 0)
            
            # for _ in range(kk):
            for i in range(len(args.etypes)):
                logger.info(f"Training CL:{args.etypes[i]}")
                for _ in range(kk):
                    loss = train_moco(
                        kk*epoch+_,
                        train_moco_dataloader[i],
                        model,
                        model_ema,
                        contrast[i],
                        criterion,
                        optimizer,
                        sw,
                        args,
                        args.etypes[i]
                    )
                saving_model(model, model_ema, optimizer, epoch, contrast, sw)
        if args.linkpred:
            k=1
            if args.moco: 
                k=5
            for i in range(len(args.lp_etypes)):
                logger.info(f"LINK PREDICT TURN:{args.lp_etypes[i]}")
                for _ in range(k):
                    loss = train_lp2(
                        k*epoch+_,
                        train_lp_dataloader[i],
                        model,
                        compute_loss,
                        optimizer,
                        sw,
                        args,
                        args.lp_etypes[i]
                    )
                saving_model(model, model_ema if args.moco else None, optimizer, epoch, contrast, sw)
        time2 = time.time()
        logger.info("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

def saving_model(model, model_ema, optimizer, epoch, contrast, sw):
    # saving the model
    logger.info("==> Saving...")
    state = {
        "opt": args,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    if args.moco:
        try:
            state["contrast"] = contrast.state_dict()
        except:
            state["contrast"] = [i.state_dict() for i in contrast]
        state["model_ema"] = model_ema.state_dict()
    save_file = os.path.join(args.model_folder, "current.pth")
    torch.save(state, save_file)
    if epoch % args.save_freq == 0:
        save_file = os.path.join(
            args.model_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
        )
        torch.save(state, save_file)
        torch.save(args, os.path.join(args.model_folder, "args"))
    sw.flush()
    # help release GPU memory
    del state
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    CUDA_MEMCHECK_PATCH_MODULE=1
    warnings.simplefilter("once", UserWarning)
    args = parse_option()

    # # For test
    # args.version = 'test'
    # args.moco = False 
    # args.linkpred = True
    # args.moco_type = '0'
    # args.gpu = 1
    # args.lp_type = '2'


    args.model_path = os.path.join(args.output_path,args.version,'model')
    args.tb_path = os.path.join(args.output_path,args.version,'tb')

    # in bounds central node is etype[-1]
    etypes = [('emoji', 'ein', 'post'), ('post', 'hasw', 'word'), ('word', 'withe', 'emoji')]
    metapaths=[['hase', 'ein'], ['win', 'hasw'], ['by', 'withe']]
    args.etypes = [etypes[int(i)] for i in args.moco_type.split(',')]
    args.metapaths=[metapaths[int(i)] for i in args.moco_type.split(',')]
    args.lp_etypes = [etypes[int(i)] for i in args.lp_type.split(',')]
    args.lp_metapaths=[metapaths[int(i)] for i in args.lp_type.split(',')]
    # args.batch_size2 = 64
    main(args)
