"""Morgan/ECFP fingerprint MLP baseline. A genuine structure-only inductive
baseline -- fingerprints exist for any drug with a SMILES string, cold-start
or not -- trained/evaluated through the exact same train.py/evaluate.py
harness as DDIModel, via the same model_factory + batch_fn contract. No
special-casing in train.py for this to work.
"""
import json
import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models import Head

log = logging.getLogger("baselines")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINGERPRINT_CACHE_PATH = os.path.join(ROOT, "data", "fingerprint_cache.pt")

FINGERPRINT_BITS = 1024
FINGERPRINT_RADIUS = 2


def smiles_to_fingerprint(smiles: str) -> Optional[np.ndarray]:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS)
    return np.array(fp, dtype=np.float32)


class FingerprintCache:
    """Disk-cached id -> Morgan/ECFP bit vector. Mirrors GraphCache/TextCache
    in data_pipeline.py: builds only what's missing, persists the rest."""

    def __init__(self, cache_path: str = FINGERPRINT_CACHE_PATH):
        self.cache_path = cache_path
        self.cache: Dict[str, np.ndarray] = {}
        if os.path.exists(cache_path):
            self.cache = torch.load(cache_path, weights_only=False)
            log.info(f"loaded fingerprint cache: {len(self.cache)} drugs")

    def build_or_load(self, smiles_map: pd.DataFrame) -> Dict[str, np.ndarray]:
        missing = smiles_map[~smiles_map["id"].isin(self.cache.keys())]
        n_failed = 0
        for _, row in missing.iterrows():
            fp = smiles_to_fingerprint(row["smiles"])
            if fp is None:
                n_failed += 1
                log.warning(f"failed to fingerprint SMILES for {row['id']}: {row['smiles']!r}")
                continue
            self.cache[row["id"]] = fp
        if len(missing) > 0:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            torch.save(self.cache, self.cache_path)
            log.info(f"fingerprint cache: built {len(missing) - n_failed} new fingerprints, "
                     f"{n_failed} failures, {len(self.cache)} total cached")
        return self.cache


class FingerprintMLP(nn.Module):
    """Same pair-representation convention as DDIModel ([a, b, a*b]) and the
    same Head class, but with a plain MLP fingerprint encoder instead of a
    graph/text branch. forward(fp_a, fp_b) matches the 2-tensor batch_fn
    contract so train.py needs no special-casing for this model."""

    def __init__(self, fp_dim: int = FINGERPRINT_BITS, hidden_dim: int = 64, head_hidden: int = 128,
                 num_classes: Optional[int] = None, binary: bool = False, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = Head(hidden_dim * 3, hidden_dim=head_hidden, num_classes=num_classes,
                          binary=binary, dropout=dropout)

    def forward(self, fp_a: torch.Tensor, fp_b: torch.Tensor) -> torch.Tensor:
        ea = self.encoder(fp_a)
        eb = self.encoder(fp_b)
        pair = torch.cat([ea, eb, ea * eb], dim=-1)
        return self.head(pair)


def build_fingerprint_batch(pairs, fp_cache: Dict[str, np.ndarray], device: str):
    fp_a = torch.stack([torch.tensor(fp_cache[a], dtype=torch.float) for a, _ in pairs]).to(device)
    fp_b = torch.stack([torch.tensor(fp_cache[b], dtype=torch.float) for _, b in pairs]).to(device)
    return (fp_a, fp_b)
