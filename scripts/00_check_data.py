"""Confirm raw DDI data downloads and parses; report dataset stats and a
PubChem drug-text coverage estimate. No modeling here -- this is a gate
before building the real pipeline. See section 3 / 12 of the research brief.
"""
import io
import logging
import os
import random
import sys
import zipfile

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("check_data")

REPO_RAW = "https://raw.githubusercontent.com/kanz76/GMPNN-CS/master"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DRUGBANK_PATH = os.path.join(DATA_DIR, "drugbank.tab")
TWOSIDES_ZIP_PATH = os.path.join(DATA_DIR, "twosides_ge_500.zip")
PUBCHEM_DESC_CACHE_HINT = os.path.join(DATA_DIR, "text_cache.json")

TEXT_SAMPLE_SIZE = 60


def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        log.info(f"cached: {dest} ({os.path.getsize(dest):,} bytes)")
        return
    log.info(f"downloading {url} -> {dest}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(r.content)
    log.info(f"saved {len(r.content):,} bytes")


def load_drugbank() -> pd.DataFrame:
    download(f"{REPO_RAW}/data/drugbank.tab", DRUGBANK_PATH)
    df = pd.read_csv(DRUGBANK_PATH, sep="\t")
    required = {"ID1", "ID2", "Y", "X1", "X2"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"drugbank.tab missing expected columns: {missing}; got {list(df.columns)}")
    return df


def load_twosides_sample() -> pd.DataFrame:
    download(f"{REPO_RAW}/data/twosides_ge_500.zip", TWOSIDES_ZIP_PATH)
    with zipfile.ZipFile(TWOSIDES_ZIP_PATH) as zf:
        names = zf.namelist()
        log.info(f"twosides zip contains: {names}")
        csv_name = next((n for n in names if n.lower().endswith((".csv", ".tab", ".tsv"))), names[0])
        with zf.open(csv_name) as f:
            sep = "\t" if csv_name.lower().endswith((".tab", ".tsv")) else ","
            df = pd.read_csv(io.BytesIO(f.read()), sep=sep)
    return df


def smiles_to_inchikey(smiles: str):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)


def pubchem_description(inchikey: str):
    """Return a short text description for a compound via PubChem PUG-REST, or None."""
    try:
        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/cids/JSON"
        r = requests.get(cid_url, timeout=10)
        if r.status_code != 200:
            return None
        cids = r.json().get("IdentifierList", {}).get("CID", [])
        if not cids:
            return None
        cid = cids[0]
        desc_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/description/JSON"
        r2 = requests.get(desc_url, timeout=10)
        if r2.status_code != 200:
            return None
        info = r2.json().get("InformationList", {}).get("Information", [])
        for item in info:
            if "Description" in item:
                return item["Description"]
        return None
    except requests.RequestException:
        return None


def main():
    log.info("=== DrugBank multi-relational DDI dataset ===")
    db = load_drugbank()
    n_pairs = len(db)
    n_drugs = len(set(db["ID1"]) | set(db["ID2"]))
    n_types = db["Y"].nunique()
    log.info(f"pairs: {n_pairs:,}")
    log.info(f"unique drugs: {n_drugs:,}")
    log.info(f"interaction types: {n_types}")
    log.info(f"type distribution (top 5): {db['Y'].value_counts().head(5).to_dict()}")

    smiles_map = pd.concat([
        db[["ID1", "X1"]].rename(columns={"ID1": "id", "X1": "smiles"}),
        db[["ID2", "X2"]].rename(columns={"ID2": "id", "X2": "smiles"}),
    ]).drop_duplicates(subset="id")
    log.info(f"unique drug SMILES resolved from pair table: {len(smiles_map):,}")

    log.info("=== TWOSIDES binary robustness dataset ===")
    try:
        ts = load_twosides_sample()
        log.info(f"twosides rows: {len(ts):,}, columns: {list(ts.columns)}")
    except Exception as e:
        log.warning(f"twosides load failed (non-fatal, will retry in pipeline step): {e}")

    log.info(f"=== PubChem text-coverage sample (n={TEXT_SAMPLE_SIZE} drugs) ===")
    random.seed(0)
    sample_ids = random.sample(list(smiles_map["id"]), min(TEXT_SAMPLE_SIZE, len(smiles_map)))
    sample = smiles_map[smiles_map["id"].isin(sample_ids)]

    hits, misses, no_inchikey = 0, 0, 0
    for _, row in sample.iterrows():
        key = smiles_to_inchikey(row["smiles"])
        if key is None:
            no_inchikey += 1
            continue
        desc = pubchem_description(key)
        if desc:
            hits += 1
        else:
            misses += 1

    n = len(sample)
    log.info(f"InChIKey parse failures: {no_inchikey}/{n}")
    log.info(f"PubChem description hits: {hits}/{n} ({100*hits/n:.1f}%)")
    log.info(f"PubChem description misses: {misses}/{n}")
    est_minutes = n_drugs * 2 / 4 / 60  # ~2 requests/drug, ~4 req/s budget
    log.info(f"estimated full-fetch time for all {n_drugs} drugs: ~{est_minutes:.0f} min (rate-limited, cached)")

    log.info("=== SUMMARY ===")
    print(f"drugs={n_drugs} pairs={n_pairs} interaction_types={n_types} "
          f"text_coverage_sample={hits}/{n} ({100*hits/n:.1f}%)")


if __name__ == "__main__":
    sys.exit(main())
