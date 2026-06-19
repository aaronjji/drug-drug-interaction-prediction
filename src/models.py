"""Graph encoder, frozen-text encoder, configurable fusion, and prediction head.

graph-only and text-only ablations are NOT separate model classes -- they are
DDIModel with use_graph/use_text flags, so every ablation shares identical
Fusion/Head code and only differs by which encoder branch is instantiated.
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, GINEConv, global_add_pool, global_mean_pool

log = logging.getLogger("models")


class GraphEncoder(nn.Module):
    """GIN (GINEConv, uses bond features) or GCN (edge_index only), 2-4 layers,
    sum/mean readout -> a single structural embedding per molecule."""

    def __init__(self, atom_dim: int, edge_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 3, conv_type: str = "gin", readout: str = "mean"):
        super().__init__()
        assert conv_type in ("gin", "gcn")
        assert readout in ("mean", "sum")
        self.conv_type = conv_type
        self.readout = readout
        self.input_proj = nn.Linear(atom_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            if conv_type == "gin":
                mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim))
                self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch: Batch) -> torch.Tensor:
        h = self.input_proj(batch.x)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, batch.edge_index, batch.edge_attr) if self.conv_type == "gin" else conv(h, batch.edge_index)
            h = F.relu(bn(h))
        pool = global_mean_pool if self.readout == "mean" else global_add_pool
        pooled = pool(h, batch.batch)
        return self.out_proj(pooled)


class TextEncoder(nn.Module):
    """Looks up the cached frozen-LM embedding. Missing-text drugs get a
    learned missing-token vector (not the real embedding, not garbage), and a
    binary has_text flag is concatenated so the model can learn to discount
    the placeholder rather than treat it as a real description."""

    def __init__(self, lm_dim: int, out_dim: int):
        super().__init__()
        self.missing_token = nn.Parameter(torch.zeros(lm_dim))
        self.proj = nn.Linear(lm_dim + 1, out_dim)

    def forward(self, cached_embedding: torch.Tensor, has_text: torch.Tensor) -> torch.Tensor:
        mask = has_text.bool().unsqueeze(-1)
        token = self.missing_token.unsqueeze(0).expand_as(cached_embedding)
        emb = torch.where(mask, cached_embedding, token)
        flag = has_text.float().unsqueeze(-1)
        return self.proj(torch.cat([emb, flag], dim=-1))


class Fusion(nn.Module):
    """concat / gated / cross_attention, switchable by config. If only one
    branch is active this is a pass-through (graph-only / text-only ablation)."""

    def __init__(self, dim: int, fusion_type: str, use_graph: bool, use_text: bool):
        super().__init__()
        assert fusion_type in ("concat", "gated", "cross_attention")
        assert use_graph or use_text
        self.fusion_type = fusion_type
        self.use_graph = use_graph
        self.use_text = use_text
        self.both = use_graph and use_text

        if not self.both:
            self.out_dim = dim
            return

        if fusion_type == "concat":
            self.out_dim = dim * 2
        elif fusion_type == "gated":
            self.gate = nn.Linear(dim * 2, dim)
            self.out_dim = dim
        elif fusion_type == "cross_attention":
            self.graph_to_text = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
            self.text_to_graph = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
            self.out_dim = dim * 2

    def forward(self, graph_emb: Optional[torch.Tensor], text_emb: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.both:
            return graph_emb if self.use_graph else text_emb

        if self.fusion_type == "concat":
            return torch.cat([graph_emb, text_emb], dim=-1)

        if self.fusion_type == "gated":
            gate = torch.sigmoid(self.gate(torch.cat([graph_emb, text_emb], dim=-1)))
            return gate * graph_emb + (1 - gate) * text_emb

        # cross_attention: each modality attends to the other (seq_len=1 tokens)
        g = graph_emb.unsqueeze(1)
        t = text_emb.unsqueeze(1)
        g_attn, _ = self.graph_to_text(g, t, t)
        t_attn, _ = self.text_to_graph(t, g, g)
        return torch.cat([g_attn.squeeze(1), t_attn.squeeze(1)], dim=-1)


class Head(nn.Module):
    """MLP over the fused pair representation. Binary (BioSNAP) or
    multi-class interaction-type (DrugBank) logits."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_classes: Optional[int] = None,
                 binary: bool = False, dropout: float = 0.1):
        super().__init__()
        assert binary or num_classes is not None
        out_dim = 1 if binary else num_classes
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DDIModel(nn.Module):
    """Full model: per-drug GraphEncoder + TextEncoder -> Fusion, then a pair
    head over [fused_a, fused_b, fused_a * fused_b]. use_graph/use_text select
    the ablation; fusion_type selects the fusion config -- both are one-line
    config switches, no architecture forking."""

    def __init__(self, atom_dim: int, edge_dim: int, lm_dim: int, hidden_dim: int = 64,
                 fusion_type: str = "concat", use_graph: bool = True, use_text: bool = True,
                 graph_layers: int = 3, graph_conv: str = "gin", graph_readout: str = "mean",
                 num_classes: Optional[int] = None, binary: bool = False, head_hidden: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        assert use_graph or use_text, "at least one branch must be active"
        self.use_graph = use_graph
        self.use_text = use_text

        if use_graph:
            self.graph_encoder = GraphEncoder(atom_dim, edge_dim, hidden_dim, hidden_dim,
                                               num_layers=graph_layers, conv_type=graph_conv,
                                               readout=graph_readout)
        if use_text:
            self.text_encoder = TextEncoder(lm_dim, hidden_dim)

        self.fusion = Fusion(hidden_dim, fusion_type, use_graph, use_text)
        fused_dim = self.fusion.out_dim
        pair_dim = fused_dim * 3  # [a, b, a*b]
        self.head = Head(pair_dim, hidden_dim=head_hidden, num_classes=num_classes,
                          binary=binary, dropout=dropout)

    def encode_drug(self, graph_batch: Optional[Batch], text_emb: Optional[torch.Tensor],
                     has_text: Optional[torch.Tensor]) -> torch.Tensor:
        g = self.graph_encoder(graph_batch) if self.use_graph else None
        t = self.text_encoder(text_emb, has_text) if self.use_text else None
        return self.fusion(g, t)

    def forward(self, graph_a: Optional[Batch], text_a: Optional[torch.Tensor], has_text_a: Optional[torch.Tensor],
                graph_b: Optional[Batch], text_b: Optional[torch.Tensor], has_text_b: Optional[torch.Tensor]
                ) -> torch.Tensor:
        ea = self.encode_drug(graph_a, text_a, has_text_a)
        eb = self.encode_drug(graph_b, text_b, has_text_b)
        pair = torch.cat([ea, eb, ea * eb], dim=-1)
        return self.head(pair)


class NegativeSampler:
    """Seeded uniform negative sampling for a positives-only binary DDI dataset
    (BioSNAP ChCh-Miner has no labeled negatives). Samples drug pairs not present
    in the positive set, in either order, excluding self-pairs.

    Negative ratio is reviewer-sensitive for positives-only graphs (it sets the
    effective class balance and the difficulty -- uniform random negatives are
    usually easy positives-vs-random-noise, not hard near-misses); the ratio is
    therefore an explicit, logged, seeded constructor argument, default 1:1.
    """

    def __init__(self, drug_ids, positive_pairs, seed: int = 0, ratio: int = 1):
        self.drug_ids = list(drug_ids)
        self.positive_set = {frozenset(p) for p in positive_pairs}
        self.ratio = ratio
        self.rng = __import__("random").Random(seed)
        log.info(f"NegativeSampler: {len(self.drug_ids)} drugs, {len(self.positive_set)} positives, "
                 f"ratio={ratio}:1 negative:positive, seed={seed}")

    def sample(self, n_positives: int):
        n_needed = n_positives * self.ratio
        negatives = []
        seen = set()
        attempts = 0
        max_attempts = n_needed * 50 + 1000
        while len(negatives) < n_needed and attempts < max_attempts:
            attempts += 1
            a, b = self.rng.choice(self.drug_ids), self.rng.choice(self.drug_ids)
            if a == b:
                continue
            key = frozenset((a, b))
            if key in self.positive_set or key in seen:
                continue
            seen.add(key)
            negatives.append((a, b))
        if len(negatives) < n_needed:
            log.warning(f"NegativeSampler: only found {len(negatives)}/{n_needed} negatives after {attempts} attempts")
        return negatives
