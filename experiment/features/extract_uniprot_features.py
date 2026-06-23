"""Fetch UniProt features for BorodaTM 62 proteins and save as .npy.

Usage: python extract_uniprot_features.py
Output: X_uniprot.npy (546, 53)
"""
import json
import os
import pickle
import time
import urllib.request
from collections import Counter

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(BASE_DIR, 'metadata.pkl')
CACHE_DIR = os.path.join(BASE_DIR, 'uniprot_cache')
OUTPUT_PATH = os.path.join(BASE_DIR, 'X_uniprot.npy')

UNIPROT_API = 'https://rest.uniprot.org/uniprotkb/{}.json'

os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_uniprot(uniprot_id):
    cache_file = os.path.join(CACHE_DIR, f'{uniprot_id}.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    try:
        req = urllib.request.Request(
            UNIPROT_API.format(uniprot_id),
            headers={'User-Agent': 'Python/3', 'Accept': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        print(f'  Fetched: {uniprot_id}')
        return data
    except Exception as e:
        print(f'  FAILED {uniprot_id}: {e}')
        return None


def extract_features(data):
    feats = {}
    if data is None:
        return feats

    # Basic
    seq = data.get('sequence', {})
    feats['length'] = seq.get('length', 0) or 0
    feats['mol_weight'] = seq.get('molWeight', 0) or 0
    feats['annotation_score'] = data.get('annotationScore', 0) or 0

    # Protein existence
    pe_map = {'Uncertain': 0, 'Predicted': 1, 'Inferred from homology': 2,
              'Evidence at transcript level': 3, 'Evidence at protein level': 4}
    feats['existence_level'] = pe_map.get(data.get('proteinExistence', ''), 0)

    # Sequence features
    feat_counts = Counter(f.get('type', 'unknown') for f in data.get('features', []))
    for key in ['Transmembrane', 'Topological domain', 'Signal peptide', 'Disulfide bond',
                'Glycosylation', 'Mutagenesis', 'Sequence conflict', 'Natural variant',
                'Helix', 'Beta strand', 'Turn', 'Active site', 'Binding site',
                'Metal binding', 'Modified residue', 'Repeat', 'Motif', 'Coiled coil',
                'Zinc finger', 'DNA binding', 'Intramembrane']:
        feats[f'n_{key.lower().replace(" ", "_")}'] = feat_counts.get(key, 0)
    feats['total_features'] = sum(feat_counts.values())

    # GO Terms
    go_counts = Counter()
    for ref in data.get('uniProtKBCrossReferences', []):
        if ref.get('database') == 'GO':
            for prop in ref.get('properties', []):
                if prop.get('key') == 'GoTerm':
                    val = prop.get('value', '')
                    go_counts[val[0] if val else '?'] += 1
    feats['go_cellular_component'] = go_counts.get('C', 0)
    feats['go_molecular_function'] = go_counts.get('F', 0)
    feats['go_biological_process'] = go_counts.get('P', 0)
    feats['go_total'] = sum(go_counts.values())

    # Cross-references
    xref_counts = Counter(ref.get('database', '') for ref in data.get('uniProtKBCrossReferences', []))
    for db in ['Pfam', 'InterPro', 'SMART', 'PROSITE', 'PDB', 'STRING', 'Reactome', 'KEGG', 'EMBL']:
        feats[f'xref_{db.lower()}'] = xref_counts.get(db, 0)
    feats['xref_total'] = sum(xref_counts.values())

    # Disease
    comments = data.get('comments', [])
    feats['n_disease_associations'] = sum(1 for c in comments if c.get('commentType') == 'DISEASE')
    feats['n_interactions'] = sum(
        len(c.get('interactions', [])) for c in comments if c.get('commentType') == 'INTERACTION')

    # Subcellular
    sub_locs = []
    for c in comments:
        if c.get('commentType') == 'SUBCELLULAR LOCATION':
            for loc in c.get('subcellularLocations', []):
                sub_locs.append(loc.get('location', {}).get('value', ''))
    feats['n_subcellular_locations'] = len(sub_locs)
    feats['membrane_location_count'] = sum(1 for loc in sub_locs if any(
        kw in loc.lower() for kw in ['membrane', 'transmembrane', 'plasma', 'er', 'golgi']))

    # Protein name keywords
    full_name = data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
    for kw in ['receptor', 'channel', 'transporter']:
        feats[f'name_contains_{kw}'] = int(kw in full_name.lower())
    feats['name_contains_g_protein'] = int('g protein' in full_name.lower())

    # Keywords
    keywords = [kw.get('name', '') for kw in data.get('keywords', [])]
    feats['n_keywords'] = len(keywords)
    kw_lower = ' '.join(keywords).lower()
    for kw in ['transmembrane', 'membrane', 'transport', 'disease']:
        feats[f'kw_{kw}'] = int(kw in kw_lower)

    return feats


def main():
    # Load metadata
    with open(META_PATH, 'rb') as f:
        meta = pickle.load(f)
    uniprot_ids = meta['uniprot_ids']  # list of 546 UniProt IDs
    unique_ids = sorted(set(uniprot_ids))
    print(f'Unique proteins: {len(unique_ids)}')

    # Fetch all
    cache = {}
    for uid in unique_ids:
        cache[uid] = fetch_uniprot(uid)
        time.sleep(0.1)  # rate limit

    # Collect feature keys
    all_keys = set()
    for data in cache.values():
        all_keys.update(extract_features(data).keys())

    # Sort and build feature dict per protein
    sorted_keys = sorted(all_keys)
    protein_feats = {}
    for uid in unique_ids:
        feats = extract_features(cache[uid])
        protein_feats[uid] = [feats.get(k, 0) for k in sorted_keys]

    # Map to each mutation
    X = np.array([protein_feats[uid] for uid in uniprot_ids], dtype=np.float32)
    print(f'Built X_uniprot: {X.shape}')
    print(f'Feature keys ({len(sorted_keys)}): {sorted_keys}')

    np.save(OUTPUT_PATH, X)
    print(f'Saved to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
