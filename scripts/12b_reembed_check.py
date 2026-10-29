"""Standalone check behind the paper's re-embedding claim (Appendix E): embeds
descriptions that the strict leakage filter leaves unchanged with the same frozen
PubMedBERT + pooling as data/text_embeddings.pt and reports the max |diff| to the
cached vectors. scripts/12 runs the same check only when it builds a new
embedding file; this script records it to results/reembed_check.json either way.
No training, no files other than the JSON are written.
"""
import json
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import data_pipeline as dp  # noqa: E402


def main(n=20):
    with open(os.path.join(ROOT, "configs", "data.yaml")) as f:
        data_cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "data", "text_cache.json")) as f:
        text_cache = json.load(f)
    with open(os.path.join(ROOT, "data", "text_cache_filtered.json"), encoding="utf-8") as f:
        filtered = json.load(f)
    original = torch.load(dp.TEXT_EMBED_CACHE_PATH, weights_only=False)
    # same selection as scripts/12: first n kept descriptions the filter left unchanged
    unchanged = [k for k, v in filtered.items() if v and v == text_cache[k]][:n]
    embedder = dp.TextEmbedder(data_cfg["text_model_name"])
    re_vecs = embedder.embed([text_cache[k] for k in unchanged])
    diffs = [float(np.abs(np.asarray(re_vecs[i]) - np.asarray(original["embeddings"][k])).max())
             for i, k in enumerate(unchanged)]
    out = {"n_descriptions": len(unchanged), "max_abs_diff": max(diffs), "median_abs_diff": float(np.median(diffs)),
           "model": data_cfg["text_model_name"], "device": "cuda" if torch.cuda.is_available() else "cpu"}
    with open(os.path.join(ROOT, "results", "reembed_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
