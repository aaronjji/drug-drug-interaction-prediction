"""Text-leakage screen + filtered-text artifacts (camera-ready, reviewers SM1A
and tFTz): do the PubChem descriptions state interaction-relevant
pharmacology that maps onto DrugBank's templated interaction types, letting
the text branch look the label up rather than predict it?

1. Screen. Every description is split into sentences and matched against a
   lexicon with two parts: (a) LABEL_TERMS -- the effect term of every one of
   the 86 DrugBank templates (data/drugbank.tab, 'Map' column), with its
   morphological variants and the pharmacological-class names that define
   it (e.g. 'anticoagul' for "...may increase the anticoagulant activities of
   #Drug2", 'platelet' for the antiplatelet types), each tagged with the
   template ids it derives from; (b) INTERACTION_TERMS -- generic interaction
   and pharmacokinetic-mechanism wording (co-administration, CYP450,
   P-glycoprotein, serum/plasma concentration, ...). Reported: how many
   descriptions/sentences match each part, and, per evaluation split, how
   many test pairs have a description that contains a term of the pair's own
   interaction type (the direct "look-up" route).
2. Filter. Every matching sentence is removed (deliberately aggressive: a
   ChEBI "It has a role as ..." sentence goes entirely if any listed role
   matches). A description left empty is treated as missing text, i.e. the
   model's learned missing-text token, exactly like drugs without a PubChem
   description. The kept text is re-embedded with the same frozen PubMedBERT
   and mean pooling as data_pipeline.TextEmbedder.

No model is trained here; 13_camera_ready_experiments.py --exps filtered-text
retrains fusion=concat and text-only on these artifacts.

Writes data/text_cache_filtered.json, data/text_embeddings_filtered.pt,
results/text_leakage_screen.{json,md}.
"""
import json
import logging
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import data_pipeline as dp
import splits as sp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("text_leakage_screen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# regex stem -> 0-indexed DrugBank interaction types (see data/drugbank.tab 'Map')
LABEL_TERMS = {
    # pharmacodynamic "...the X activities of #Drug2" and "risk or severity of X" types
    r"photosensiti": [0],
    r"cholinergic|muscarinic": [1],
    r"vasoconstrict|vasopressor|sympathomimetic": [4, 18, 83],
    r"anticoagul|vitamin k|thromb|fibrinoly": [5, 11, 80],
    r"platelet": [25, 52],
    r"bleed|h(?:a)?emorrh": [65],
    r"toxic|toxin": [6, 14, 34, 43, 56, 57, 79],
    r"glyc(?:a)?emi|insulin|diabet": [8, 77],
    r"hyperten|hypotens|blood pressure": [9, 29, 30, 36, 40, 55, 59, 70, 78],
    r"bronch": [13, 45],
    r"depressant|depression|sedati|hypnotic|anxiolytic|an(?:a)?esthe|anticonvulsant|antiepilep"
    r"|opioid|narcotic|barbiturat|benzodiazepin|gaba": [15, 31, 39, 40, 75, 78],
    r"neuromuscular|muscle relaxant": [16, 20, 21],
    r"stimula|neuroexcit": [22, 26, 60],
    r"calc(?:a)?emi": [23, 50],
    r"atrioventricular|\bav block|arrhythm|bradycard|tachycard|\bqtc?\b|torsade|heart failure"
    r"|channel blocker": [19, 24, 32, 49, 53, 81, 84],
    r"dermatolog": [27],
    r"diuretic": [28],
    r"immunosuppress|myelosuppress": [33, 62],
    r"antipsychotic|neuroleptic": [35],
    r"vasodilat": [37],
    r"constipat|laxative": [38],
    r"respiratory depress": [39],
    r"kal(?:a)?emi": [41, 67, 82],
    r"natr(?:a)?emi": [54],
    r"myopath|rhabdomyoly": [47],
    r"analges|anti-inflammatory|nsaid": [51, 68],
    r"ulcer": [58],
    r"serotonin|serotonerg|\b5-ht": [63],
    r"fluid retention|fluid retaining|\bo?edema": [73],
    r"hypersensitiv|allerg|anaphyla": [85],
    r"adrenergic": [4, 9, 13, 18, 36, 53, 70, 84],
    r"diagnostic|contrast agent|radio-?opaque|imaging agent": [44],
    r"adverse|side[- ]effect": [17, 21, 27, 48],
    r"efficacy|effectiveness": [7, 10, 44, 64, 66, 69],
    # pharmacokinetic types (metabolism, serum concentration, absorption, ...)
    r"metaboli[sz](?:m|e|ed|es|ing)\b|active metabolite": [3, 10, 46, 76],
    r"bioavailab": [2, 61],
    r"absor[bp]": [12, 17, 66],
    r"serum|plasma (?:concentration|level)": [10, 17, 64, 66, 71, 72, 74, 76],
    r"protein[- ]bin|protein[- ]bound": [42],
    r"excret|eliminat|clearance|half-life": [64, 71],
}
INTERACTION_TERMS = (r"interact|co-?administ|coadminist|concomitan|combination|combined with|in conjunction with"
                     r"|together with|contraindicat|potentiat|\bcyp(?:\d|\b)|cytochrome|p450|p-?glycoprotein|\bp-?gp\b"
                     r"|enzyme induc|inducer")

LABEL_RX = {pat: re.compile(pat, re.I) for pat in LABEL_TERMS}
INTERACTION_RX = re.compile(INTERACTION_TERMS, re.I)
TERMS_BY_TYPE = {}
for pat, types in LABEL_TERMS.items():
    for t in types:
        TERMS_BY_TYPE.setdefault(t, []).append(LABEL_RX[pat])

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def sentences(text):
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def label_hit(sentence):
    return any(rx.search(sentence) for rx in LABEL_RX.values())


def interaction_hit(sentence):
    return bool(INTERACTION_RX.search(sentence))


def removes(sentence, level):
    """level 'all': interaction-type terms OR generic interaction wording;
    level 'generic': generic interaction wording only (keeps pharmacological
    roles such as 'anticoagulant', removes explicit co-administration/CYP/
    P-gp/contraindication statements)."""
    return interaction_hit(sentence) or (level == "all" and label_hit(sentence))


def filter_description(text, level="all"):
    kept = [s for s in sentences(text) if not removes(s, level)]
    return " ".join(kept)


def type_term_match(text, interaction_type):
    return bool(text) and any(rx.search(text) for rx in TERMS_BY_TYPE.get(interaction_type, []))


def pair_match_rates(df, text_cache):
    """Share of pairs whose interaction type has >=1 type-specific lexicon term
    and whose #Drug2 (ID2) / either drug's description contains one of them."""
    has_terms = df["Y"].isin(TERMS_BY_TYPE.keys())
    d2 = np.array([type_term_match(text_cache.get(b), y) for b, y in zip(df["ID2"], df["Y"])])
    d1 = np.array([type_term_match(text_cache.get(a), y) for a, y in zip(df["ID1"], df["Y"])])
    return {"n_pairs": int(len(df)), "share_types_with_terms": float(has_terms.mean()),
            "share_drug2_desc_matches_own_type": float(d2.mean()),
            "share_either_desc_matches_own_type": float((d1 | d2).mean())}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=("all", "generic"), default="all",
                        help="'all' (default, the paper's filtered text) or 'generic' (explicit interaction "
                             "wording only); 'generic' writes *_generic files")
    level = parser.parse_args().level
    suffix = "" if level == "all" else f"_{level}"
    with open(os.path.join(ROOT, "configs", "data.yaml")) as f:
        data_cfg = yaml.safe_load(f)
    with open(dp.TEXT_CACHE_PATH, encoding="utf-8") as f:
        text_cache = json.load(f)
    docs = {k: v for k, v in text_cache.items() if v}
    log.info(f"{len(docs)}/{len(text_cache)} drugs have a PubChem description")

    templates = pd.read_csv(dp.DRUGBANK_PATH, sep="\t").assign(Y=lambda d: d["Y"] - 1) \
        .groupby("Y")["Map"].first()
    missing_types = sorted(set(range(dp.NUM_INTERACTION_TYPES)) - set(TERMS_BY_TYPE))
    log.info(f"interaction types without a type-specific lexicon term (PK/boilerplate-only "
             f"wording is covered by the generic terms): {missing_types}")

    # --- 1. screen ---
    n_sent = Counter()
    doc_label = doc_inter = 0
    per_term_docs = Counter()
    for text in docs.values():
        sents = sentences(text)
        lab = [label_hit(s) for s in sents]
        inter = [interaction_hit(s) for s in sents]
        n_sent["total"] += len(sents)
        n_sent["label"] += sum(lab)
        n_sent["interaction"] += sum(inter)
        n_sent["either"] += sum(a or b for a, b in zip(lab, inter))
        n_sent["removed"] += sum(removes(s, level) for s in sents)
        doc_label += any(lab)
        doc_inter += any(inter)
        for pat, rx in LABEL_RX.items():
            per_term_docs[pat] += bool(rx.search(text))

    # --- 2. filter ---
    filtered = {k: (filter_description(v, level) if v else None) for k, v in text_cache.items()}
    filtered = {k: (v if v else None) for k, v in filtered.items()}
    words_before = sum(len(v.split()) for v in docs.values())
    words_after = sum(len(filtered[k].split()) for k in docs if filtered[k])
    emptied = [k for k in docs if not filtered[k]]
    changed = [k for k in docs if filtered[k] != docs[k]]
    residual = [k for k, v in filtered.items() if v and any(removes(s, level) for s in sentences(v))]
    assert not residual, f"filter left matching sentences in {residual[:5]}"

    transductive, drug_split, inductive = sp.load_splits(0, tag="drugbank")
    split_frames = {"transductive_train": transductive[transductive["split"] == "train"],
                    "transductive_test": transductive[transductive["split"] == "test"],
                    "inductive_train": inductive["train"], "s1": inductive["s1"], "s2": inductive["s2"]}
    match_full = {n: pair_match_rates(df, text_cache) for n, df in split_frames.items()}
    match_filtered = {n: pair_match_rates(df, filtered) for n, df in split_frames.items()}
    cov = {g: float(np.mean([bool(filtered.get(i)) for i in ids])) for g, ids in drug_split.items()}
    cov_full = {g: float(np.mean([bool(text_cache.get(i)) for i in ids])) for g, ids in drug_split.items()}

    stats = {
        "n_drugs": len(text_cache), "n_with_description": len(docs), "n_sentences": n_sent["total"],
        "docs_with_label_term": doc_label, "docs_with_interaction_term": doc_inter,
        "sentences_with_label_term": n_sent["label"], "sentences_with_interaction_term": n_sent["interaction"],
        "sentences_removed": n_sent["removed"], "share_sentences_removed": n_sent["removed"] / n_sent["total"],
        "words_before": words_before, "words_after": words_after,
        "share_words_removed": 1 - words_after / words_before,
        "docs_changed": len(changed), "docs_emptied": len(emptied),
        "n_with_description_after_filter": len(docs) - len(emptied),
        "text_coverage_by_inductive_drug_group_full": cov_full,
        "text_coverage_by_inductive_drug_group_filtered": cov,
        "pair_lexical_match_full_text": match_full, "pair_lexical_match_filtered_text": match_filtered,
        "docs_per_label_term": dict(per_term_docs.most_common()),
        "types_without_type_specific_term": missing_types,
        "lexicon": {"label_terms": LABEL_TERMS, "interaction_terms": INTERACTION_TERMS},
    }
    log.info(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}, indent=1))

    with open(os.path.join(ROOT, "data", f"text_cache_filtered{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(filtered, f)

    # --- 3. embed the kept text with the same frozen LM + pooling as the original cache ---
    # Reuses an existing embedding file built from this exact filtered text (GPU
    # re-embedding differs at ~1e-5, and trained filtered-text runs must keep
    # pointing at the vectors they were trained on).
    keep_ids = [k for k, v in filtered.items() if v]
    embed_path = os.path.join(ROOT, "data", f"text_embeddings_filtered{suffix}.pt")
    existing = torch.load(embed_path, weights_only=False) if os.path.exists(embed_path) else None
    if existing is not None and existing.get("texts") == {k: filtered[k] for k in keep_ids}:
        log.info(f"reusing {embed_path}: built from this exact filtered text")
    else:
        embedder = dp.TextEmbedder(data_cfg["text_model_name"])
        vecs = embedder.embed([filtered[k] for k in keep_ids])
        original = torch.load(dp.TEXT_EMBED_CACHE_PATH, weights_only=False)
        assert original["dim"] == embedder.dim
        # sanity: re-embedding an UNCHANGED description must reproduce the cached vector
        unchanged = [k for k in keep_ids if filtered[k] == text_cache[k]][:20]
        if unchanged:
            re_vecs = embedder.embed([text_cache[k] for k in unchanged])
            max_diff = float(max(np.abs(re_vecs[i] - original["embeddings"][k]).max() for i, k in enumerate(unchanged)))
            stats["reembed_check_max_abs_diff_unchanged_texts"] = max_diff
            log.info(f"re-embedding check on {len(unchanged)} unchanged descriptions: max |diff| = {max_diff:.2e}")
            assert max_diff < 1e-3, "re-embedding does not reproduce the cached vectors -- different LM/pooling?"
        torch.save({"embeddings": dict(zip(keep_ids, vecs)), "dim": embedder.dim,
                    "texts": {k: filtered[k] for k in keep_ids},
                    "source": f"text_cache_filtered{suffix}.json", "level": level, "model": data_cfg["text_model_name"]}, embed_path)
        log.info(f"embedded {len(keep_ids)} filtered descriptions")

    with open(os.path.join(ROOT, "results", f"text_leakage_screen{suffix}.json"), "w") as f:
        json.dump(stats, f, indent=2)

    examples = [k for k in changed if filtered[k]][:4] + emptied[:2]
    lines = [
        f"# Text-leakage screen (camera-ready), filter level: {level}", "",
        f"- Drugs with a PubChem description: {len(docs)}/{len(text_cache)}; {n_sent['total']} sentences.",
        f"- Descriptions with generic co-administration / interaction-mechanism wording (combination use, "
        f"CYP450, P-glycoprotein, enzyme induction, contraindication, ...): {doc_inter} ({doc_inter / len(docs):.1%}).",
        f"- Descriptions with >=1 interaction-type term (template effect terms + class synonyms): "
        f"{doc_label} ({doc_label / len(docs):.1%}).",
        f"- Filter removed {n_sent['removed']} sentences ({n_sent['removed'] / n_sent['total']:.1%}) and "
        f"{1 - words_after / words_before:.1%} of words; {len(changed)} descriptions changed, "
        f"{len(emptied)} emptied (-> missing-text token); {len(docs) - len(emptied)} drugs keep some text.",
        "", "## Pairs whose own interaction type's term appears in a description", "",
        "| split | n pairs | types with a term | Drug2 desc. matches (full) | either (full) | either (filtered) |",
        "|---|---|---|---|---|---|",
    ]
    for n in split_frames:
        a, b = match_full[n], match_filtered[n]
        lines.append(f"| {n} | {a['n_pairs']} | {a['share_types_with_terms']:.1%} | "
                     f"{a['share_drug2_desc_matches_own_type']:.1%} | {a['share_either_desc_matches_own_type']:.1%} | "
                     f"{b['share_either_desc_matches_own_type']:.1%} |")
    lines += ["", "## Text coverage by inductive drug group (full -> filtered)", ""]
    lines += [f"- {g}: {cov_full[g]:.1%} -> {cov[g]:.1%}" for g in cov]
    lines += ["", "## Examples (before -> after)", ""]
    for k in examples:
        lines += [f"**{k}**", "", f"- before: {docs[k]}", f"- after: {filtered[k] or '(empty -> missing-text token)'}", ""]
    with open(os.path.join(ROOT, "results", f"text_leakage_screen{suffix}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info("wrote results/text_leakage_screen.{json,md}, data/text_cache_filtered.json, "
             "data/text_embeddings_filtered.pt")


if __name__ == "__main__":
    main()
