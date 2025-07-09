#!/usr/bin/env python
# encoding: utf-8

import os
import torch
import torch.nn as nn
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import dgl

from gcc.models import GraphEncoder
from gcc.datasets.graph_dataset_new import NodeClassificationDatasetv2
from gcc.datasets.data_util import batcher

logger = logging.getLogger(__name__)


class BaseDownstreamTask(ABC):
    """
    Base class for downstream tasks using the pre-trained GNN checkpoint.
    Handles checkpoint loading, embedding generation, and common utilities.
    """
    
    def __init__(
        self, 
        checkpoint_path: str,
        device: str = 'cuda:0',
        dgl_graphs_file: Optional[str] = None
    ):
        """
        Initialize the base downstream task.
        
        Args:
            checkpoint_path: Path to the pre-trained checkpoint
            device: Device to run on ('cpu' or 'cuda:x')
            dgl_graphs_file: Path to DGL graph file for inference
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.dgl_graphs_file = dgl_graphs_file
        
        # Load checkpoint and model
        self.checkpoint_args, self.model = self._load_checkpoint()
        
        # Task-specific components (to be implemented by subclasses)
        self.task_head = None
        
    def _load_checkpoint(self) -> Tuple[Any, nn.Module]:
        """Load the pre-trained GNN checkpoint."""
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")
            
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        args = checkpoint["opt"]
        
        # Create model with same architecture as training
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
        
        # Load pre-trained weights
        model.load_state_dict(checkpoint["model"])
        model = model.to(self.device)
        model.eval()  # Set to evaluation mode
        
        logger.info(f"Model loaded successfully (epoch {checkpoint['epoch']})")
        del checkpoint
        
        return args, model
    
    def _create_dataloader(
        self, 
        etype: Tuple[str, str, str],
        metapath: List[str],
        batch_size: int = 32,
        shuffle: bool = False
    ) -> torch.utils.data.DataLoader:
        """Create a dataloader for the specified edge type and metapath."""
        if self.dgl_graphs_file is None:
            raise ValueError("dgl_graphs_file must be provided to create dataloader")
            
        dataset = NodeClassificationDatasetv2(
            dgl_graphs_file=self.dgl_graphs_file,
            rw_hops=self.checkpoint_args.rw_hops,
            subgraph_size=self.checkpoint_args.subgraph_size,
            restart_prob=self.checkpoint_args.restart_prob,
            positional_embedding_size=self.checkpoint_args.positional_embedding_size,
            device=self.device,
            etype=etype,
            metapath=metapath
        )
        
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=batcher(),
            shuffle=shuffle,
        )
    
    def generate_embeddings(
        self, 
        etype: Tuple[str, str, str],
        metapath: List[str],
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Generate embeddings for nodes of the specified type.
        
        Args:
            etype: Edge type tuple (src, rel, dst)
            metapath: Metapath for sampling
            batch_size: Batch size for processing
            
        Returns:
            Node embeddings tensor
        """
        dataloader = self._create_dataloader(etype, metapath, batch_size, shuffle=False)
        
        embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                graph_q, graph_k = batch
                bsz = graph_q.batch_size
                graph_q = graph_q.to(self.device)
                graph_k = graph_k.to(self.device)
                
                # Get embeddings from pre-trained model
                feat_q = self.model(graph_q, etype)
                feat_k = self.model(graph_k, etype)
                
                # Average the two views
                emb = (feat_q + feat_k) / 2
                embeddings.append(emb.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def extract_node_embeddings(
        self,
        graph: dgl.DGLGraph,
        etype: Tuple[str, str, str]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract node embeddings for all node types from a graph.
        
        Args:
            graph: Input DGL graph
            etype: Edge type for processing
            
        Returns:
            Dictionary mapping node types to their embeddings
        """
        graph = graph.to(self.device)
        
        with torch.no_grad():
            # Get encoder outputs (all layer representations)
            h_reps = self.model.encode(graph, etype)
            
            # Use the last layer representation
            node_embeddings = h_reps[-1]
            
            # If heterogeneous graph, split by node type
            if hasattr(graph, 'ntypes') and len(graph.ntypes) > 1:
                embeddings_by_type = {}
                start_idx = 0
                for ntype in graph.ntypes:
                    num_nodes = graph.num_nodes(ntype)
                    embeddings_by_type[ntype] = node_embeddings[start_idx:start_idx + num_nodes]
                    start_idx += num_nodes
                return embeddings_by_type
            else:
                return {'default': node_embeddings}
    
    @abstractmethod
    def setup_task_head(self, **kwargs):
        """Setup the task-specific prediction head. To be implemented by subclasses."""
        pass
    
    @abstractmethod
    def train(self, **kwargs):
        """Train the downstream task. To be implemented by subclasses."""
        pass
    
    @abstractmethod
    def evaluate(self, **kwargs):
        """Evaluate the downstream task. To be implemented by subclasses."""
        pass
    
    def save_model(self, save_path: str):
        """Save the task-specific components."""
        if self.task_head is not None:
            torch.save({
                'task_head': self.task_head.state_dict(),
                'checkpoint_args': self.checkpoint_args
            }, save_path)
            logger.info(f"Task model saved to {save_path}")
    
    def load_task_model(self, model_path: str):
        """Load the task-specific components."""
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Task model not found at {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        if self.task_head is not None:
            self.task_head.load_state_dict(checkpoint['task_head'])
        logger.info(f"Task model loaded from {model_path}")