"""Wires a named architecture to a (model_factory, batch_fn) pair consumable
by train.train_one_run. The only place that knows about DDIModel/FingerprintMLP
specifics -- train.py itself stays fully model-agnostic.
"""
from typing import Callable, Dict, Tuple

import torch.nn as nn

import data_pipeline as dp
from baselines import FingerprintMLP, build_fingerprint_batch
from models import DDIModel

DDI_ARCH_FLAGS = {
    "fusion=concat": dict(fusion_type="concat", use_graph=True, use_text=True),
    "fusion=gated": dict(fusion_type="gated", use_graph=True, use_text=True),
    "fusion=cross_attention": dict(fusion_type="cross_attention", use_graph=True, use_text=True),
    "graph-only": dict(fusion_type="concat", use_graph=True, use_text=False),
    "text-only": dict(fusion_type="concat", use_graph=False, use_text=True),
}


def make_ddi_components(arch_name: str, common: dict, num_classes, binary: bool,
                         graphs, text_emb_cache, has_text, lm_dim: int, device: str
                         ) -> Tuple[Callable[[], nn.Module], Callable]:
    flags = DDI_ARCH_FLAGS[arch_name]
    kwargs = {**common, **flags}

    def model_factory():
        return DDIModel(num_classes=num_classes, binary=binary, **kwargs)

    def batch_fn(pairs, dev):
        return dp.build_pair_batch(pairs, graphs, text_emb_cache, has_text, lm_dim, dev)

    return model_factory, batch_fn


def make_fingerprint_components(fp_cache: Dict, fp_dim: int, hidden_dim: int, head_hidden: int,
                                 dropout: float, num_classes, binary: bool
                                 ) -> Tuple[Callable[[], nn.Module], Callable]:
    def model_factory():
        return FingerprintMLP(fp_dim=fp_dim, hidden_dim=hidden_dim, head_hidden=head_hidden,
                               num_classes=num_classes, binary=binary, dropout=dropout)

    def batch_fn(pairs, dev):
        return build_fingerprint_batch(pairs, fp_cache, dev)

    return model_factory, batch_fn
