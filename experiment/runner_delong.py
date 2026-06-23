"""DeLong test for statistical significance of AUC differences between methods.

Compares:
  A. Feature ablation: 250D vs +UniProt vs +ESM3 vs +ESM3+UniProt
  B. Our models vs external tools (SIFT, PolyPhen-2, PROVEAN, FATHMM)

All comparisons use the same CV splits (paired design) for valid DeLong testing.
"""

import os, sys, json, time, warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, RepeatedStratifiedKFold
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
ESM3_79 = os.path.join(FEAT_DIR, 'esm3_79', 'X_esm3_79.npy')
UNIPROT = os.path.join(FEAT_DIR, 'X_uniprot.npy')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def delong_roc_test(y_true, scores_a, scores_b):
    """DeLong test for two correlated ROC curves.

    y_true: ground truth labels (0/1)
    scores_a, scores_b: prediction scores for two methods on the same samples

    Returns: (auc_a, auc_b, z_stat, p_value)
    """
    y = np.asarray(y_true).astype(int)
    s_a = np.asarray(scores_a, dtype=float)
    s_b = np.asarray(scores_b, dtype=float)

    n = len(y)
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)

    if n_pos == 0 or n_neg == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')

    # Compute AUC via Mann-Whitney U statistic components
    # For each positive-negative pair, compute the structural components
    # V_{10} matrix for each method, then covariance

    # Placeholder values
    pos_scores_a = s_a[y == 1]
    neg_scores_a = s_a[y == 0]
    pos_scores_b = s_b[y == 1]
    neg_scores_b = s_b[y == 0]

    # Compute AUC
    auc_a = _auc_from_scores(pos_scores_a, neg_scores_a)
    auc_b = _auc_from_scores(pos_scores_b, neg_scores_b)

    if n_pos < 2 or n_neg < 2:
        return auc_a, auc_b, 0.0, 1.0

    # DeLong's variance-covariance computation
    # Structural components for each positive and negative observation
    # V^X_{10}(x_k) for positive samples (k=1..n_pos)
    # V^X_{01}(x_k) for negative samples (k=1..n_neg)

    def compute_v10(scores, pos_idx, neg_idx):
        """Compute V_10 for positive samples (structural component)."""
        pos_s = scores[pos_idx]
        neg_s = scores[neg_idx]
        return np.array([np.mean(neg_s < p + 0.5 * (neg_s == p)) for p in pos_s])

    def compute_v01(scores, pos_idx, neg_idx):
        """Compute V_01 for negative samples."""
        pos_s = scores[pos_idx]
        neg_s = scores[neg_idx]
        return np.array([np.mean(pos_s > n + 0.5 * (pos_s == n)) for n in neg_s])

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    v10_a = compute_v10(s_a, pos_idx, neg_idx)
    v10_b = compute_v10(s_b, pos_idx, neg_idx)
    v01_a = compute_v01(s_a, pos_idx, neg_idx)
    v01_b = compute_v01(s_b, pos_idx, neg_idx)

    # Variance of the difference
    # Using the formula from DeLong et al. (1988)
    s10 = np.cov(v10_a, v10_b, ddof=1) if len(pos_idx) > 2 else np.cov(v10_a, v10_b)
    s01 = np.cov(v01_a, v01_b, ddof=1) if len(neg_idx) > 2 else np.cov(v01_a, v01_b)

    var_auc_a = s10[0, 0] / n_pos + s01[0, 0] / n_neg
    var_auc_b = s10[1, 1] / n_pos + s01[1, 1] / n_neg
    cov_auc_ab = s10[0, 1] / n_pos + s01[0, 1] / n_neg

    var_diff = var_auc_a + var_auc_b - 2 * cov_auc_ab
    var_diff = max(var_diff, 1e-12)

    z_stat = (auc_a - auc_b) / np.sqrt(var_diff)
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    return auc_a, auc_b, z_stat, p_value


def _auc_from_scores(pos_scores, neg_scores):
    """Compute AUC from positive and negative scores using Mann-Whitney U."""
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    # Count pairs where positive > negative
    U = 0
    for p in pos_scores:
        U += np.sum(p > neg_scores) + 0.5 * np.sum(p == neg_scores)
    return U / (n_pos * n_neg)


# ---- Model factories ----
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
    X_tr = X_tr.astype(np.float64); X_te = X_te.astype(np.float64)
    y_tr = y_tr.astype(int)
    if k < X_tr.shape[1]:
        idx = select_top_k(X_tr, y_tr, k)
        X_tr = X_tr[:, idx]; X_te = X_te[:, idx]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)
    all_probas = []
    for factory, s in models:
        m = factory(s); m.fit(X_tr_s, y_tr)
        all_probas.append(m.predict_proba(X_te_s)[:, 1])
    return np.mean(all_probas, axis=0)


def get_tool_score(name, scores_dict, indices):
    """Get properly directed tool score for AUC computation."""
    s = scores_dict[name][indices]
    # Negate if lower = pathogenic (so AUC > 0.5)
    if name in ('SIFT', 'PROVEAN', 'FATHMM', 'ESM-1v-zero-shot'):
        return -s
    return s


def main():
    from scipy.stats import norm as scipy_norm

    print(f"{'#'*70}")
    print(f"# DeLong Test: Statistical Significance of AUC Differences")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    # ---- Load data ----
    print("Loading data...")
    df = load_data(); df = remove_separator_column(df)
    y_all = map_labels(df.iloc[:, 1])
    proteins = df.iloc[:, 0].apply(lambda x: str(x).split('_')[0]).values
    X_250d_df, _ = get_features_and_labels(df)
    X_250d = pd.DataFrame(X_250d_df.values, index=X_250d_df.index)

    esm3 = np.load(ESM3_79)
    uniprot = np.load(UNIPROT)
    X_B = pd.DataFrame(np.hstack([X_250d.values, esm3]), index=X_250d.index)
    X_C = pd.DataFrame(np.hstack([X_250d.values, uniprot]), index=X_250d.index)
    X_D = pd.DataFrame(np.hstack([X_250d.values, esm3, uniprot]), index=X_250d.index)

    # Load tool scores
    tool_scores = {}
    for col, name in [('SIFT_SCORE', 'SIFT'), ('pph2_prob', 'PolyPhen-2'),
                       ('PROVEAN_SCORE', 'PROVEAN'), ('fathmm_Score', 'FATHMM')]:
        if col in df.columns:
            tool_scores[name] = df[col].values.astype(float)

    # Load ESM-1v zero-shot
    zs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'improved_results', 'esm_zeroshot_scores.npy')
    if os.path.exists(zs_path):
        tool_scores['ESM-1v-zero-shot'] = np.load(zs_path)

    # Our model configurations
    our_configs = {
        '250D Baseline': (X_250d, None),
        '250D+UniProt': (X_C, None),
        '250D+ESM3': (X_B, 1024),
        '250D+ESM3+UniProt': (X_D, 1024),
    }

    models_to_use = [(make_xgb, SEED)]
    if HAS_LGB:
        models_to_use.append((make_lgb, SEED))
    if HAS_CAT:
        models_to_use.append((make_cat, SEED))

    # ================================================================
    # Part A: DeLong on 10-fold CV (paired predictions per fold)
    # ================================================================
    print(f"\n{'='*70}")
    print("PART A: DeLong Test on 10-fold CV Predictions")
    print(f"{'='*70}")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    # Collect all OOF predictions from each method
    method_names = list(our_configs.keys()) + list(tool_scores.keys())
    oof_predictions = {name: np.zeros(len(y_all)) for name in method_names}
    oof_y_true = np.zeros(len(y_all))

    for tr, te in skf.split(X_250d, y_all):
        y_tr = y_all.iloc[tr].values.astype(int)
        y_te = y_all.iloc[te].values.astype(int)
        oof_y_true[te] = y_te

        # Our models
        for name, (X_data, k) in our_configs.items():
            ys = ensemble_predict_fold(
                X_data.iloc[tr].values, X_data.iloc[te].values, y_tr,
                k or 99999, models_to_use)
            oof_predictions[name][te] = ys

        # External tools (no training needed, just use scores directly)
        for name in tool_scores:
            oof_predictions[name][te] = tool_scores[name][te]

    # Run DeLong test for all pairs
    def run_delong_matrix(names, preds_dict, y_true, label=""):
        n = len(names)
        results = []
        for i in range(n):
            for j in range(i + 1, n):
                name_a, name_b = names[i], names[j]
                # For tool scores, use properly directed scores
                s_a = preds_dict[name_a]
                s_b = preds_dict[name_b]
                # For AUC computation, ensure correct direction
                if name_a in ('SIFT', 'PROVEAN', 'FATHMM', 'ESM-1v-zero-shot'):
                    s_a_for_auc = -s_a
                else:
                    s_a_for_auc = s_a
                if name_b in ('SIFT', 'PROVEAN', 'FATHMM', 'ESM-1v-zero-shot'):
                    s_b_for_auc = -s_b
                else:
                    s_b_for_auc = s_b

                auc_a, auc_b, z_stat, p_val = delong_roc_test(y_true, s_a_for_auc, s_b_for_auc)
                sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
                results.append({
                    'method_a': name_a, 'method_b': name_b,
                    'auc_a': auc_a, 'auc_b': auc_b,
                    'delta': auc_a - auc_b, 'z_stat': z_stat, 'p_value': p_val,
                    'significant': sig,
                })
                print(f"  {name_a:<28} vs {name_b:<28}  "
                      f"ΔAUC={auc_a-auc_b:+.4f}  Z={z_stat:+.3f}  p={p_val:.4f}  {sig}")
        return results

    print("\n--- OUR MODELS: Feature Ablation ---")
    our_names = list(our_configs.keys())
    our_delong = run_delong_matrix(our_names, oof_predictions, oof_y_true)

    print("\n--- OUR BEST vs EXTERNAL TOOLS ---")
    best_name = '250D+ESM3+UniProt'
    tool_names = list(tool_scores.keys())
    all_names = [best_name] + tool_names
    vs_tools = run_delong_matrix(all_names, oof_predictions, oof_y_true)

    print("\n--- EXTERNAL TOOLS: Among Themselves ---")
    tool_delong = run_delong_matrix(tool_names, oof_predictions, oof_y_true)

    # ================================================================
    # Part B: DeLong on GroupKFold (cross-protein)
    # ================================================================
    print(f"\n{'='*70}")
    print("PART B: DeLong Test on GroupKFold Predictions")
    print(f"{'='*70}")

    gkf = GroupKFold(n_splits=10)
    gk_predictions = {name: np.zeros(len(y_all)) for name in method_names}
    gk_y_true = np.zeros(len(y_all))

    for tr, te in gkf.split(X_250d, y_all, proteins):
        y_tr = y_all.iloc[tr].values.astype(int)
        y_te = y_all.iloc[te].values.astype(int)
        gk_y_true[te] = y_te

        for name, (X_data, k) in our_configs.items():
            ys = ensemble_predict_fold(
                X_data.iloc[tr].values, X_data.iloc[te].values, y_tr,
                k or 99999, models_to_use)
            gk_predictions[name][te] = ys

        for name in tool_scores:
            gk_predictions[name][te] = tool_scores[name][te]

    print("\n--- OUR MODELS: Feature Ablation (GroupKFold) ---")
    gk_our = run_delong_matrix(our_names, gk_predictions, gk_y_true)

    print("\n--- OUR BEST vs EXTERNAL TOOLS (GroupKFold) ---")
    gk_vs_tools = run_delong_matrix(all_names, gk_predictions, gk_y_true)

    # ================================================================
    # Summary Tables
    # ================================================================
    print(f"\n{'#'*70}")
    print("# SUMMARY: DeLong Test Results")
    print(f"{'#'*70}")

    # Feature ablation summary
    print(f"\n{'Feature Ablation (10-fold CV OOF predictions)':^70}")
    print(f"{'Comparison':<55} {'ΔAUC':>8} {'p-value':>8}")
    print("-" * 72)
    for r in our_delong:
        print(f"  {r['method_a']:<26} vs {r['method_b']:<26} {r['delta']:>+7.4f} {r['p_value']:>8.4f}")

    print(f"\n{'Our Best vs External Tools (10-fold CV)':^70}")
    print(f"{'Comparison':<55} {'ΔAUC':>8} {'p-value':>8}")
    print("-" * 72)
    for r in vs_tools:
        print(f"  {r['method_a']:<26} vs {r['method_b']:<26} {r['delta']:>+7.4f} {r['p_value']:>8.4f}")

    print(f"\n{'Feature Ablation (GroupKFold)':^70}")
    print(f"{'Comparison':<55} {'ΔAUC':>8} {'p-value':>8}")
    print("-" * 72)
    for r in gk_our:
        print(f"  {r['method_a']:<26} vs {r['method_b']:<26} {r['delta']:>+7.4f} {r['p_value']:>8.4f}")

    print(f"\n{'Our Best vs External Tools (GroupKFold)':^70}")
    print(f"{'Comparison':<55} {'ΔAUC':>8} {'p-value':>8}")
    print("-" * 72)
    for r in gk_vs_tools:
        print(f"  {r['method_a']:<26} vs {r['method_b']:<26} {r['delta']:>+7.4f} {r['p_value']:>8.4f}")

    # Save
    all_delong = {
        'our_models_10fold': our_delong,
        'vs_tools_10fold': vs_tools,
        'tools_among_themselves': tool_delong,
        'our_models_gk': gk_our,
        'vs_tools_gk': gk_vs_tools,
    }
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(RESULTS_DIR, f'delong_{timestamp}.json')
    with open(out_path, 'w') as f:
        json.dump(all_delong, f, indent=2)
    print(f"\nSaved to {out_path}")

    # ================================================================
    # Final Summary Matrix for Paper
    # ================================================================
    print(f"\n{'#'*70}")
    print("# PAPER-READY SUMMARY")
    print(f"{'#'*70}")

    print(f"\n{'Method':<30} {'AUC (10f)':>10} {'vs Baseline':>10} {'p':>8} {'sig':>6}")
    print("-" * 70)
    # Baseline AUC
    bl_auc = None
    for r in our_delong:
        if (r['method_a'] == '250D Baseline' and '250D+UniProt' in r['method_b']) or \
           (r['method_b'] == '250D Baseline' and r['method_a'] == '250D+UniProt'):
            if r['method_a'] == '250D Baseline':
                bl_auc = r['auc_a']
            else:
                bl_auc = r['auc_b']
            break
    if bl_auc is None and our_delong:
        bl_auc = our_delong[0]['auc_a'] if our_delong[0]['method_a'] == '250D Baseline' else our_delong[0]['auc_b']

    if bl_auc:
        print(f"  {'250D Baseline':<28} {bl_auc:>10.4f} {'—':>10} {'—':>8} {'—':>6}")

    for r in our_delong:
        if r['method_a'] == '250D Baseline' and r['method_b'] != '250D Baseline':
            other = r['method_b']; auc = r['auc_b']; delta = r['auc_b'] - r['auc_a']; p = r['p_value']; sig = r['significant']
            print(f"  {other:<28} {auc:>10.4f} {delta:>+10.4f} {p:>8.4f} {sig:>6}")
        elif r['method_b'] == '250D Baseline' and r['method_a'] != '250D Baseline':
            other = r['method_a']; auc = r['auc_a']; delta = r['auc_a'] - r['auc_b']; p = r['p_value']; sig = r['significant']
            print(f"  {other:<28} {auc:>10.4f} {delta:>+10.4f} {p:>8.4f} {sig:>6}")

    print("-" * 70)
    for r in vs_tools:
        if r['method_a'] == best_name and r['method_b'] in tool_names:
            print(f"  {r['method_b']:<28} {r['auc_b']:>10.4f} {r['auc_b']-r['auc_a']:>+10.4f} {r['p_value']:>8.4f} {r['significant']:>6}")
        elif r['method_b'] == best_name and r['method_a'] in tool_names:
            print(f"  {r['method_a']:<28} {r['auc_a']:>10.4f} {r['auc_a']-r['auc_b']:>+10.4f} {r['p_value']:>8.4f} {r['significant']:>6}")

    print(f"\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")


if __name__ == '__main__':
    main()
