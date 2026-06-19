"""SMILES -> molecular graph, drug text fetch + cache, frozen-LM text embedding cache.

All heavy steps (graph construction, PubChem text fetch, LM embedding) are
cached to disk under data/ so they run once. Nothing here trains a model.
"""
import io
import json
import logging
import os
import time
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from torch_geometric.data import Batch, Data

log = logging.getLogger("data_pipeline")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

DRUGBANK_PATH = os.path.join(DATA_DIR, "drugbank.tab")
CHCH_MINER_PATH = os.path.join(DATA_DIR, "chch_miner.tsv.gz")
GRAPH_CACHE_PATH = os.path.join(DATA_DIR, "graph_cache.pt")
TEXT_CACHE_PATH = os.path.join(DATA_DIR, "text_cache.json")
TEXT_EMBED_CACHE_PATH = os.path.join(DATA_DIR, "text_embeddings.pt")

# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

def _download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    log.info(f"downloading {url} -> {dest}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(r.content)


NUM_INTERACTION_TYPES = 86  # Y is zero-indexed 0..85 from this loader onward


def load_drugbank_pairs(repo_raw: str) -> pd.DataFrame:
    """Multi-relational DDI pairs: columns ID1, ID2, Y (interaction type, 0..85), X1, X2 (SMILES).
    drugbank.tab itself ships Y as 1..86; shifted to 0-indexed here, once, at the
    source, so every downstream consumer (splits, evaluable-class sets, training
    targets) agrees -- a prior version converted ad hoc at each call site and
    that mismatch silently broke evaluable-class lookups."""
    _download(f"{repo_raw}/data/drugbank.tab", DRUGBANK_PATH)
    df = pd.read_csv(DRUGBANK_PATH, sep="\t")
    df["Y"] = df["Y"] - 1
    return df


def drug_smiles_table(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Unique drug id -> SMILES, derived from the pair table (ID1/X1, ID2/X2)."""
    table = pd.concat([
        pairs_df[["ID1", "X1"]].rename(columns={"ID1": "id", "X1": "smiles"}),
        pairs_df[["ID2", "X2"]].rename(columns={"ID2": "id", "X2": "smiles"}),
    ]).drop_duplicates(subset="id").reset_index(drop=True)
    return table


def load_chch_miner_binary(url: str, smiles_map: pd.DataFrame) -> pd.DataFrame:
    """Binary DDI dataset (BioSNAP ChCh-Miner). Has no SMILES of its own, so we
    restrict to drugs we already have SMILES for (via drugbank.tab) and document
    the drop. Returns columns d1, d2 (both positive/interacting pairs)."""
    _download(url, CHCH_MINER_PATH)
    chch = pd.read_csv(CHCH_MINER_PATH, sep="\t", header=None, compression="gzip", names=["d1", "d2"])
    has_smiles = set(smiles_map["id"])
    before = len(chch)
    filtered = chch[chch["d1"].isin(has_smiles) & chch["d2"].isin(has_smiles)].reset_index(drop=True)
    log.info(f"ChCh-Miner: kept {len(filtered):,}/{before:,} pairs after restricting to drugs with known SMILES")
    return filtered


# ---------------------------------------------------------------------------
# SMILES -> PyG molecular graph
# ---------------------------------------------------------------------------

ATOM_VOCAB = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se", "Unknown"]
DEGREE_VOCAB = [0, 1, 2, 3, 4, 5]
CHARGE_VOCAB = [-2, -1, 0, 1, 2]
NUM_H_VOCAB = [0, 1, 2, 3, 4]


def _one_hot(value, vocab) -> List[float]:
    vec = [0.0] * len(vocab)
    if value in vocab:
        vec[vocab.index(value)] = 1.0
    else:
        vec[-1] = 1.0
    return vec


def _atom_features(atom) -> List[float]:
    from rdkit import Chem
    hybridization_vocab = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    feats = []
    feats += _one_hot(atom.GetSymbol(), ATOM_VOCAB)
    feats += _one_hot(atom.GetDegree(), DEGREE_VOCAB)
    feats += _one_hot(atom.GetFormalCharge(), CHARGE_VOCAB)
    feats += _one_hot(atom.GetHybridization(), hybridization_vocab)
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    feats += _one_hot(atom.GetTotalNumHs(), NUM_H_VOCAB)
    return feats


ATOM_FEATURE_DIM = len(ATOM_VOCAB) + len(DEGREE_VOCAB) + len(CHARGE_VOCAB) + 5 + 1 + len(NUM_H_VOCAB)


def _bond_features(bond) -> List[float]:
    from rdkit import Chem
    bond_type_vocab = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    feats = _one_hot(bond.GetBondType(), bond_type_vocab)
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    return feats


BOND_FEATURE_DIM = 4 + 1 + 1


def smiles_to_pyg_data(smiles: str) -> Optional[Data]:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = _bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [feat, feat]
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class GraphCache:
    """Disk-cached id -> PyG Data. Only builds graphs for ids not already cached."""

    def __init__(self, cache_path: str = GRAPH_CACHE_PATH):
        self.cache_path = cache_path
        self.cache: Dict[str, Data] = {}
        if os.path.exists(cache_path):
            self.cache = torch.load(cache_path, weights_only=False)
            log.info(f"loaded graph cache: {len(self.cache)} drugs")

    def build_or_load(self, smiles_map: pd.DataFrame) -> Dict[str, Data]:
        missing = smiles_map[~smiles_map["id"].isin(self.cache.keys())]
        n_failed = 0
        for _, row in missing.iterrows():
            data = smiles_to_pyg_data(row["smiles"])
            if data is None:
                n_failed += 1
                log.warning(f"failed to parse SMILES for {row['id']}: {row['smiles']!r}")
                continue
            self.cache[row["id"]] = data
        if len(missing) > 0:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            torch.save(self.cache, self.cache_path)
            log.info(f"graph cache: built {len(missing) - n_failed} new graphs, {n_failed} parse failures, "
                      f"{len(self.cache)} total cached")
        return self.cache


# ---------------------------------------------------------------------------
# Drug text: PubChem fetch + cache
# ---------------------------------------------------------------------------

def smiles_to_inchikey(smiles: str) -> Optional[str]:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    key = Chem.MolToInchiKey(mol)
    return key or None


def _pubchem_description(inchikey: str, max_retries: int = 3) -> Optional[str]:
    cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/cids/JSON"
    for attempt in range(max_retries):
        try:
            r = requests.get(cid_url, timeout=10)
            if r.status_code == 404:
                return None
            if r.status_code != 200:
                time.sleep(2 ** attempt)
                continue
            cids = r.json().get("IdentifierList", {}).get("CID", [])
            if not cids:
                return None
            cid = cids[0]
            break
        except requests.RequestException:
            time.sleep(2 ** attempt)
    else:
        return None

    desc_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/description/JSON"
    for attempt in range(max_retries):
        try:
            r = requests.get(desc_url, timeout=10)
            if r.status_code != 200:
                time.sleep(2 ** attempt)
                continue
            info = r.json().get("InformationList", {}).get("Information", [])
            for item in info:
                if "Description" in item:
                    return item["Description"]
            return None
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return None


class TextCache:
    """Disk-cached id -> description-or-None. Fetches only missing ids, rate-limited,
    saves incrementally so a crash doesn't lose progress."""

    def __init__(self, cache_path: str = TEXT_CACHE_PATH):
        self.cache_path = cache_path
        self.cache: Dict[str, Optional[str]] = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
            log.info(f"loaded text cache: {len(self.cache)} drugs")

    def _save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f)

    def build_or_load(self, smiles_map: pd.DataFrame, rate_limit_sec: float = 0.3,
                       max_retries: int = 3, save_every: int = 50) -> Dict[str, Optional[str]]:
        missing = smiles_map[~smiles_map["id"].isin(self.cache.keys())]
        n_total = len(missing)
        n_hit = 0
        for i, (_, row) in enumerate(missing.iterrows()):
            key = smiles_to_inchikey(row["smiles"])
            desc = _pubchem_description(key, max_retries=max_retries) if key else None
            self.cache[row["id"]] = desc
            if desc:
                n_hit += 1
            time.sleep(rate_limit_sec)
            if (i + 1) % save_every == 0:
                self._save()
                log.info(f"text fetch progress: {i + 1}/{n_total} ({n_hit} hits so far)")
        if n_total > 0:
            self._save()
            log.info(f"text fetch done: {n_hit}/{n_total} new hits "
                     f"({len(self.cache)} total cached, "
                     f"{sum(1 for v in self.cache.values() if v)} total with text)")
        return self.cache

    def has_text(self) -> Dict[str, bool]:
        return {k: bool(v) for k, v in self.cache.items()}


# ---------------------------------------------------------------------------
# Frozen LM text embeddings
# ---------------------------------------------------------------------------

class TextEmbedder:
    """Wraps a frozen biomedical LM. Mean-pools token embeddings over the
    attention mask. The LM never runs during training -- embeddings are
    precomputed once here and cached to disk."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        from transformers import AutoModel, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def embed(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            out.append(pooled.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim), dtype=np.float32)


def build_or_load_text_embeddings(text_cache: Dict[str, Optional[str]], model_name: str,
                                   cache_path: str = TEXT_EMBED_CACHE_PATH) -> Tuple[Dict[str, np.ndarray], int]:
    """Returns (id -> embedding) for drugs with non-empty cached text, and the embedding dim.
    Only embeds ids not already cached."""
    cached: Dict[str, np.ndarray] = {}
    dim = None
    if os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=False)
        cached = blob["embeddings"]
        dim = blob["dim"]
        log.info(f"loaded text embedding cache: {len(cached)} drugs")

    ids_with_text = [k for k, v in text_cache.items() if v]
    missing_ids = [k for k in ids_with_text if k not in cached]
    if missing_ids:
        embedder = TextEmbedder(model_name)
        dim = embedder.dim
        texts = [text_cache[k] for k in missing_ids]
        vecs = embedder.embed(texts)
        for k, v in zip(missing_ids, vecs):
            cached[k] = v
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"embeddings": cached, "dim": dim}, cache_path)
        log.info(f"embedded {len(missing_ids)} new drug descriptions (dim={dim})")
    return cached, dim


# ---------------------------------------------------------------------------
# Batch assembly (shared by model smoke tests and train.py)
# ---------------------------------------------------------------------------

def build_pair_batch(pairs: List[Tuple[str, str]], graphs: Dict[str, Data],
                      text_emb_cache: Dict[str, np.ndarray], has_text: Dict[str, bool],
                      lm_dim: int, device: str):
    """pairs: list of (id_a, id_b). Returns the two graph batches and the two
    (text_embedding, has_text) tensor pairs, aligned by position. Missing-text
    drugs get a zero placeholder here -- TextEncoder substitutes its learned
    missing-token using the has_text flag, so the placeholder's value is inert."""
    data_a = [graphs[a] for a, _ in pairs]
    data_b = [graphs[b] for _, b in pairs]
    batch_a = Batch.from_data_list(data_a).to(device)
    batch_b = Batch.from_data_list(data_b).to(device)

    def text_tensors(ids):
        emb = torch.stack([
            torch.tensor(text_emb_cache[i], dtype=torch.float) if has_text.get(i, False)
            else torch.zeros(lm_dim, dtype=torch.float)
            for i in ids
        ]).to(device)
        flags = torch.tensor([1.0 if has_text.get(i, False) else 0.0 for i in ids], device=device)
        return emb, flags

    text_a, ht_a = text_tensors([a for a, _ in pairs])
    text_b, ht_b = text_tensors([b for _, b in pairs])
    return batch_a, text_a, ht_a, batch_b, text_b, ht_b
