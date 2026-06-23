"""DeLong test for statistical significance of AUC differences — 109D baseline.

Uses 109D features (250D minus WAPSSM/PSSM columns) as the new baseline.

Compares:
  A. Feature ablation: 109D vs +UniProt vs +ESM3 vs +ESM3+UniProt
  B. Our best model vs external tools (SIFT, PolyPhen-2, PROVEAN, FATHMM)

All comparisons use the same CV splits (paired design) for valid DeLong testing.
"""

import os, sys, json, time, warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data, remove_separator_column, map_labels, get_features_and_labels

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

FEAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'features')
ESM3_79  = os.path.join(FEAT_DIR, 'esm3_79', 'X_esm3_79.npy')
UNIPROT  = os.path.join(FEAT_DIR, 'X_uniprot.npy')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ================================================================
# DeLong test
# ================================================================
def delong_roc_test(y_true, scores_a, scores_b):
    """DeLong test for two correlated ROC curves.
    Returns: (auc_a, auc_b, z_stat, p_value)
    """
    y = np.asarray(y_true).astype(int)
    s_a = np.asarray(scores_a, dtype=float)
    s_b = np.asarray(scores_b, dtype=float)

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    auc_a = _auc_from_scores(s_a[pos_idx], s_a[neg_idx])
    auc_b = _auc_from_scores(s_b[pos_idx], s_b[neg_idx])

    if n_pos < 2 or n_neg < 2:
        return auc_a, auc_b, 0.0, 1.0

    def v10(s, pi, ni):
        ps, ns = s[pi], s[ni]
        return np.array([np.mean(ns < p) + 0.5 * np.mean(ns == p) for p in ps])

    def v01(s, pi, ni):
        ps, ns = s[pi], s[ni]
        return np.array([np.mean(ps > n) + 0.5 * np.mean(ps == n) for n in ns])

    v10_a = v10(s_a, pos_idx, neg_idx)
    v10_b = v10(s_b, pos_idx, neg_idx)
    v01_a = v01(s_a, pos_idx, neg_idx)
    v01_b = v01(s_b, pos_idx, neg_idx)

    s10 = np.cov(v10_a, v10_b, ddof=1)
    s01 = np.cov(v01_a, v01_b, ddof=1)

    var_diff = (s10[0,0] + s10[1,1] - 2*s10[0,1]) / n_pos \
             + (s01[0,0] + s01[1,1] - 2*s01[0,1]) / n_neg
    var_diff = max(var_diff, 1e-12)

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return auc_a, auc_b, z, p


def _auc_from_scores(pos_scores, neg_scores):
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    U = sum(np.sum(p > neg_scores) + 0.5 * np.sum(p == neg_scores) for p in pos_scores)
    return U / (n_pos * n_neg)


# ================================================================
# Model factories
# ================================================================
def make_xgb(seed=SEED):
    return XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.9,
        objective='binary:logistic', random_state=seed,
        verbosity=0, n_jobs=4, tree_method='hist',
    )

def make_lgb(seed=SEED):
    if HAS_LGB:
        return LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbose=-1, n_jobs=4,
        )
    return None

def make_cat(seed=SEED):
    if HAS_CAT:
        return CatBoostClassifier(
            iterations=200, depth=4, learning_rate=0.1,
            random_seed=seed, verbose=0, thread_count=4,
        )
    return None


def select_top_k(X_tr, y_tr, k):
    if k >= X_tr.shape[1]:
        return np.arange(X_tr.shape[1])
    scores, _ = f_classif(X_tr, y_tr)
    return np.argsort(scores)[::-1][:k]


def ensemble_predict_fold(X_tr, X_te, y_tr, k, models):
    X_tr = X_tr.astype(np.float64)
    X_te = X_te.astype(np.float64)
    y_tr = y_tr.astype(int)
    if k < X_tr.shape[1]:
        idx = select_top_k(X_tr, y_tr, k)
        X_tr = X_tr[:, idx]; X_te = X_te[:, idx]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    probas = []
    for factory, s in models:
        m = factory(s)
        m.fit(X_tr, y_tr)
        probas.append(m.predict_proba(X_te)[:, 1])
    return np.mean(probas, axis=0)


def run_delong_matrix(names, preds_dict, y_true, negate_set=None):
    """Run DeLong test for all pairs in names. negate_set: tools where lower = pathogenic."""
    if negate_set is None:
        negate_set = set()
    results = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            sa = -preds_dict[na] if na in negate_set else preds_dict[na]
            sb = -preds_dict[nb] if nb in negate_set else preds_dict[nb]
            auc_a, auc_b, z, p = delong_roc_test(y_true, sa, sb)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            results.append({
                'method_a': na, 'method_b': nb,
                'auc_a': round(auc_a, 4), 'auc_b': round(auc_b, 4),
                'delta': round(auc_a - auc_b, 4),
                'z_stat': round(z, 3), 'p_value': round(p, 4),
                'significant': sig,
            })
            print(f"  {na:<30} vs {nb:<30}  ΔAUC={auc_a-auc_b:+.4f}  Z={z:+.3f}  p={p:.4f}  {sig}")
    return results


# ================================================================
# Main
# ================================================================
def main():
    print(f"{'#'*70}")
    print(f"# DeLong Test: 109D Baseline (no WAPSSM/PSSM)")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    # ---- Load data ----
    print("Loading data...")
    df = load_data()
    df = remove_separator_column(df)
    y_all = map_labels(df.iloc[:, 1])
    proteins = df.iloc[:, 0].apply(lambda x: str(x).split('_')[0]).values

    X_250d_df, _ = get_features_and_labels(df)
    # Drop PSSM/WAPSSM columns → 109D
    pssm_cols = [c for c in X_250d_df.columns if str(c).startswith('pssm') or 'pssm' in str(c).lower()]
    X_109d = X_250d_df.drop(columns=pssm_cols)
    print(f"  250D → drop {len(pssm_cols)} PSSM cols → 109D features: {X_109d.shape}")

    esm3    = np.load(ESM3_79)
    uniprot = np.load(UNIPROT)
    print(f"  ESM-3 L79: {esm3.shape},  UniProt: {uniprot.shape}")

    # Feature matrices
    X_A = X_109d  # 109D
    X_B = pd.DataFrame(np.hstack([X_109d.values, esm3]),    index=X_109d.index)  # +ESM3
    X_C = pd.DataFrame(np.hstack([X_109d.values, uniprot]), index=X_109d.index)  # +UniProt
    X_D = pd.DataFrame(np.hstack([X_109d.values, esm3, uniprot]), index=X_109d.index)  # +ESM3+UniProt

    our_configs = {
        '109D Baseline':             (X_A, None),
        '109D+UniProt':              (X_C, None),
        '109D+ESM3':                 (X_B, 1024),
        '109D+ESM3+UniProt':         (X_D, 1024),
    }

    # Load tool scores
    NEGATE = {'SIFT', 'PROVEAN', 'FATHMM', 'ESM-1v-zero-shot'}
    tool_scores = {}
    for col, name in [('SIFT_SCORE', 'SIFT'), ('pph2_prob', 'PolyPhen-2'),
                       ('PROVEAN_SCORE', 'PROVEAN'), ('fathmm_Score', 'FATHMM')]:
        if col in df.columns:
            tool_scores[name] = df[col].values.astype(float)
            print(f"  Loaded tool: {name}")

    zs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'improved_results', 'esm_zeroshot_scores.npy')
    if os.path.exists(zs_path):
        tool_scores['ESM-1v-zero-shot'] = np.load(zs_path)
        print(f"  Loaded tool: ESM-1v-zero-shot")

    models_to_use = [(make_xgb, SEED)]
    if HAS_LGB:
        models_to_use.append((make_lgb, SEED))
    if HAS_CAT:
        models_to_use.append((make_cat, SEED))
    print(f"  Ensemble: {[f.__name__ for f, _ in models_to_use]}")

    method_names = list(our_configs.keys()) + list(tool_scores.keys())

    # ================================================================
    # Part A: 10-fold CV
    # ================================================================
    print(f"\n{'='*70}")
    print("PART A: DeLong Test on 10-fold CV (stratified)")
    print(f"{'='*70}")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    oof_preds  = {n: np.zeros(len(y_all)) for n in method_names}
    oof_y_true = np.zeros(len(y_all))

    for fold_i, (tr, te) in enumerate(skf.split(X_A, y_all)):
        y_tr = y_all.iloc[tr].values.astype(int)
        oof_y_true[te] = y_all.iloc[te].values.astype(int)
        for name, (X_data, k) in our_configs.items():
            oof_preds[name][te] = ensemble_predict_fold(
                X_data.iloc[tr].values, X_data.iloc[te].values, y_tr,
                k or 99999, models_to_use)
        for name, scores in tool_scores.items():
            oof_preds[name][te] = scores[te]
        print(f"  Fold {fold_i+1}/10 done")

    our_names = list(our_configs.keys())
    best_name = '109D+ESM3+UniProt'
    tool_names = list(tool_scores.keys())

    print("\n--- OUR MODELS: Feature Ablation (10-fold) ---")
    our_10f = run_delong_matrix(our_names, oof_preds, oof_y_true, NEGATE)

    print("\n--- OUR BEST vs EXTERNAL TOOLS (10-fold) ---")
    vs_tools_10f = run_delong_matrix([best_name] + tool_names, oof_preds, oof_y_true, NEGATE)

    print("\n--- EXTERNAL TOOLS AMONG THEMSELVES (10-fold) ---")
    tools_10f = run_delong_matrix(tool_names, oof_preds, oof_y_true, NEGATE)

    # ================================================================
    # Part B: GroupKFold
    # ================================================================
    print(f"\n{'='*70}")
    print("PART B: DeLong Test on GroupKFold (cross-protein)")
    print(f"{'='*70}")

    gkf = GroupKFold(n_splits=10)
    gk_preds  = {n: np.zeros(len(y_all)) for n in method_names}
    gk_y_true = np.zeros(len(y_all))

    for fold_i, (tr, te) in enumerate(gkf.split(X_A, y_all, proteins)):
        y_tr = y_all.iloc[tr].values.astype(int)
        gk_y_true[te] = y_all.iloc[te].values.astype(int)
        for name, (X_data, k) in our_configs.items():
            gk_preds[name][te] = ensemble_predict_fold(
                X_data.iloc[tr].values, X_data.iloc[te].values, y_tr,
                k or 99999, models_to_use)
        for name, scores in tool_scores.items():
            gk_preds[name][te] = scores[te]
        print(f"  GK Fold {fold_i+1}/10 done")

    print("\n--- OUR MODELS: Feature Ablation (GroupKFold) ---")
    our_gk = run_delong_matrix(our_names, gk_preds, gk_y_true, NEGATE)

    print("\n--- OUR BEST vs EXTERNAL TOOLS (GroupKFold) ---")
    vs_tools_gk = run_delong_matrix([best_name] + tool_names, gk_preds, gk_y_true, NEGATE)

    # ================================================================
    # Save
    # ================================================================
    all_results = {
        'baseline': '109D (no WAPSSM)',
        'timestamp': datetime.now().isoformat(),
        'our_models_10fold': our_10f,
        'vs_tools_10fold':   vs_tools_10f,
        'tools_10fold':      tools_10f,
        'our_models_gk':     our_gk,
        'vs_tools_gk':       vs_tools_gk,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_json = os.path.join(RESULTS_DIR, f'delong_109d_{ts}.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved JSON: {out_json}")

    # ================================================================
    # Paper-ready summary
    # ================================================================
    print(f"\n{'#'*70}")
    print("# PAPER-READY SUMMARY (109D Baseline)")
    print(f"{'#'*70}")

    print(f"\n{'=== 10-fold CV: Feature Ablation ==='}")
    print(f"{'Comparison':<55} {'ΔAUC':>8} {'p':>8} {'sig':>5}")
    print("-" * 78)
    for r in our_10f:
        if r['method_b'] == '109D Baseline':
            print(f"  {r['method_a']:<52}  {r['delta']:>+8.4f} {r['p_value']:>8.4f} {r['significant']:>5}")
        elif r['method_a'] == '109D Baseline':
            print(f"  {r['method_b']:<52}  {-r['delta']:>+8.4f} {r['p_value']:>8.4f} {r['significant']:>5}")

    print(f"\n{'=== GroupKFold: Feature Ablation ==='}")
    print(f"{'Comparison':<55} {'ΔAUC':>8} {'p':>8} {'sig':>5}")
    print("-" * 78)
    for r in our_gk:
        if r['method_b'] == '109D Baseline':
            print(f"  {r['method_a']:<52}  {r['delta']:>+8.4f} {r['p_value']:>8.4f} {r['significant']:>5}")
        elif r['method_a'] == '109D Baseline':
            print(f"  {r['method_b']:<52}  {-r['delta']:>+8.4f} {r['p_value']:>8.4f} {r['significant']:>5}")

    print(f"\n{'=== Our Best vs External Tools (10-fold) ==='}")
    print(f"{'Comparison':<55} {'ΔAUC':>8} {'p':>8} {'sig':>5}")
    print("-" * 78)
    for r in vs_tools_10f:
        if r['method_a'] == best_name and r['method_b'] in tool_names:
            print(f"  {best_name} vs {r['method_b']:<20}  {r['delta']:>+8.4f} {r['p_value']:>8.4f} {r['significant']:>5}")

    print(f"\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print(f"\nAll done. Results saved to {out_json}")


if __name__ == '__main__':
    main()
