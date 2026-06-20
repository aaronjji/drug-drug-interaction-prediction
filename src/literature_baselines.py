"""SSI-DDI (Nyamabo et al. 2021, kanz76/SSI-DDI) ported into our harness.

Architecture (GAT blocks -> SAGPooling readout -> co-attention -> RESCAL) is
faithfully reproduced from the original repo (architecture verified to
reproduce a plausible test number on their OWN data/split/eval protocol --
see scripts/07_ssiddi_faithfulness_check.py). Two deliberate, documented
deviations from the original for this wrapped version:

1. Decoder: the original scores ONE given relation per example and trains via
   pos-vs-corrupted-negative sigmoid loss (a KGE-style link-prediction setup).
   We instead score ALL 86 relations per (head, tail) pair (RESCALAllRelations
   below) to get classification logits, so this drops into our standard
   softmax-cross-entropy multiclass training/eval -- identical splits,
   evaluable-class policy, and metrics as every other model in the comparison.
   The RESCAL bilinear scoring function itself (heads @ R_r @ tails^T) is
   unchanged; only the "vs one relation" -> "vs all relations" framing differs.
2. Atom features: the original computes its own 55-dim featurization. We
   reuse our existing 35-dim atom features / cached graphs (data_pipeline.py)
   instead of building a second, separate graph cache. No edge/bond features
   are used either way -- the original's GATConv calls don't consume edge_attr.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, LayerNorm, SAGPooling, global_add_pool


class SSIDDIBlock(nn.Module):
    def __init__(self, n_heads: int, in_features: int, head_out_feats: int):
        super().__init__()
        self.conv = GATConv(in_features, head_out_feats, n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)

    def forward(self, x, edge_index, batch):
        x = self.conv(x, edge_index)
        att_x, att_edge_index, _, att_batch, _, _ = self.readout(x, edge_index, batch=batch)
        global_graph_emb = global_add_pool(att_x, att_batch)
        return x, global_graph_emb


class CoAttentionLayer(nn.Module):
    """Verbatim port of the original layers.CoAttentionLayer."""

    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features // 2))
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))

    def forward(self, receiver: torch.Tensor, attendant: torch.Tensor) -> torch.Tensor:
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        return e_scores


class RESCALAllRelations(nn.Module):
    """Same bilinear scoring as the original RESCAL (heads @ R_r @ tails^T),
    but evaluated against every relation at once -> [batch, n_rels] logits
    instead of a [batch] score for one given relation. Co-attention weights
    are relation-independent (computed from heads/tails alone), so they're
    applied identically across the relation axis."""

    def __init__(self, n_rels: int, n_features: int):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, heads: torch.Tensor, tails: torch.Tensor, alpha_scores: torch.Tensor) -> torch.Tensor:
        # heads, tails: [batch, n_blocks, K]; alpha_scores: [batch, n_blocks, n_blocks]
        rels = F.normalize(self.rel_emb.weight, dim=-1).view(self.n_rels, self.n_features, self.n_features)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)
        # [batch, n_blocks_h, K] x [n_rels, K, K] x [batch, n_blocks_t, K] -> [batch, n_blocks_h, n_blocks_t, n_rels]
        scores = torch.einsum("bik,rkl,bjl->bijr", heads, rels, tails)
        scores = scores * alpha_scores.unsqueeze(-1)
        return scores.sum(dim=(1, 2))  # [batch, n_rels]


class SSIDDIModel(nn.Module):
    """forward(graph_a, graph_b) -> [batch, num_classes] logits, matching the
    same (model_factory, batch_fn) contract as DDIModel/FingerprintMLP -- no
    special-casing needed in train.py."""

    def __init__(self, atom_dim: int, num_classes: int, hidd_dim: int = 64, kge_dim: int = 64,
                 heads_out_feat_params=(32, 32, 32, 32), blocks_params=(2, 2, 2, 2)):
        super().__init__()
        assert len(heads_out_feat_params) == len(blocks_params)
        self.initial_norm = LayerNorm(atom_dim)
        self.blocks = nn.ModuleList()
        self.net_norms = nn.ModuleList()
        in_features = atom_dim
        for head_out_feats, n_heads in zip(heads_out_feat_params, blocks_params):
            self.blocks.append(SSIDDIBlock(n_heads, in_features, head_out_feats))
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads
        self.co_attention = CoAttentionLayer(kge_dim)
        self.rescal = RESCALAllRelations(num_classes, kge_dim)

    def _encode(self, batch: Batch):
        x = self.initial_norm(batch.x, batch.batch)
        edge_index, b = batch.edge_index, batch.batch
        reprs = []
        for block, norm in zip(self.blocks, self.net_norms):
            x, r = block(x, edge_index, b)
            reprs.append(r)
            x = F.elu(norm(x, b))
        return torch.stack(reprs, dim=-2)  # [batch, n_blocks, kge_dim]

    def forward(self, graph_a: Batch, graph_b: Batch) -> torch.Tensor:
        kge_heads = self._encode(graph_a)
        kge_tails = self._encode(graph_b)
        alpha = self.co_attention(kge_heads, kge_tails)
        return self.rescal(kge_heads, kge_tails, alpha)


def build_graph_only_batch(pairs, graphs, device):
    data_a = [graphs[a] for a, _ in pairs]
    data_b = [graphs[b] for _, b in pairs]
    return Batch.from_data_list(data_a).to(device), Batch.from_data_list(data_b).to(device)
