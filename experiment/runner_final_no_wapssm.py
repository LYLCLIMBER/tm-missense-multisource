"""Final ensemble experiment: XGB+LGB+Cat + ESM3 L79 + 250D + UniProt + Feature Selection.

Combines all proven strategies into one comprehensive experiment.
Feature selection (ANOVA) and standardization inside every CV fold (leak-free).

Evaluation: 10-fold CV, 10×10 CV, GroupKFold
"""
import os, sys, json, time, warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data, remove_separator_column, map_labels, get_features_and_labels

warnings.filterwarnings('ignore')
SEED = 42
SEEDS = [42, 123, 456]

FEAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'features')
ESM3_79 = os.path.join(FEAT_DIR, 'esm3_79', 'X_esm3_79.npy')
UNIPROT = os.path.join(FEAT_DIR, 'X_uniprot.npy')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_xgb(seed=SEED):
    return XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.9,
        objective='binary:logistic', random_state=seed,
        verbosity=0, n_jobs=4, tree_method='hist',
    )

def make_lgb(seed=SEED):
    return LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=seed, verbose=-1, n_jobs=4,
    )

def make_cat(seed=SEED):
    return CatBoostClassifier(
        iterations=200, depth=4, learning_rate=0.1,
        random_seed=seed, verbose=0, thread_count=4,
    )


def select_top_k(X_tr, y_tr, k):
    if k >= X_tr.shape[1]:
        return np.arange(X_tr.shape[1])
    scores, _ = f_classif(X_tr, y_tr)
    return np.argsort(scores)[::-1][:k]


def ensemble_predict_fold(X_tr_raw, X_te_raw, y_tr, k, models_to_use):
    """Train XGB+LGB+Cat and return soft-voting average predictions."""
    X_tr = X_tr_raw.astype(np.float64); X_te = X_te_raw.astype(np.float64)
    y_tr = y_tr.astype(int)

    if k < X_tr.shape[1]:
        idx = select_top_k(X_tr, y_tr, k)
        X_tr = X_tr[:, idx]; X_te = X_te[:, idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)

    all_probas = []
    for factory, s in models_to_use:
        m = factory(s); m.fit(X_tr_s, y_tr)
        all_probas.append(m.predict_proba(X_te_s)[:, 1])
    return np.mean(all_probas, axis=0)


def binary_metrics(y_true, y_score, threshold=0.5):
    """Return metrics for pathogenicity prediction, with 1 as pathogenic."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sn = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        'auc': roc_auc_score(y_true, y_score),
        'ap': average_precision_score(y_true, y_score),
        'acc': accuracy_score(y_true, y_pred),
        'bacc': balanced_accuracy_score(y_true, y_pred),
        'sn': sn,
        'sp': sp,
        'mcc': matthews_corrcoef(y_true, y_pred),
    }


def summarize_metric_dicts(metric_dicts):
    keys = metric_dicts[0].keys()
    return {
        key: {
            'mean': float(np.mean([m[key] for m in metric_dicts])),
            'std': float(np.std([m[key] for m in metric_dicts], ddof=1))
        }
        for key in keys
    }


def flatten_summary(prefix, summary):
    flat = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            flat[f'{prefix}_{key}'] = value['mean']
            flat[f'{prefix}_{key}_std'] = value['std']
        else:
            flat[f'{prefix}_{key}'] = value
    return flat


def run_10fold(X, y, k, models_to_use):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_metrics = []
    t0 = time.time()
    for tr, te in skf.split(X, y):
        ys = ensemble_predict_fold(
            X.iloc[tr].values, X.iloc[te].values, y.iloc[tr].values.astype(int),
            k, models_to_use)
        y_te = y.iloc[te].values.astype(int)
        fold_metrics.append(binary_metrics(y_te, ys))
    elapsed = time.time() - t0
    summary = summarize_metric_dicts(fold_metrics)
    summary['elapsed_sec'] = elapsed
    return summary


def run_10x10(X, y, k, models_to_use):
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=SEED)
    fold_metrics = []
    t0 = time.time()
    for tr, te in rskf.split(X, y):
        ys = ensemble_predict_fold(
            X.iloc[tr].values, X.iloc[te].values, y.iloc[tr].values.astype(int),
            k, models_to_use)
        fold_metrics.append(binary_metrics(y.iloc[te].values.astype(int), ys))
    elapsed = time.time() - t0
    summary = summarize_metric_dicts(fold_metrics)
    summary['elapsed_sec'] = elapsed
    return summary


def run_gkfold(X, y, groups, k, models_to_use):
    gkf = GroupKFold(n_splits=10)
    fold_metrics = []
    t0 = time.time()
    for tr, te in gkf.split(X, y, groups):
        ys = ensemble_predict_fold(
            X.iloc[tr].values, X.iloc[te].values, y.iloc[tr].values.astype(int),
            k, models_to_use)
        fold_metrics.append(binary_metrics(y.iloc[te].values.astype(int), ys))
    elapsed = time.time() - t0
    summary = summarize_metric_dicts(fold_metrics)
    summary['elapsed_sec'] = elapsed
    return summary


def main():
    print(f"\n{'#'*70}")
    print(f"# Final Ensemble: XGB+LGB+Cat + ESM3 L79 + 250D + UniProt")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    # ---- Load Data ----
    print("\n--- Loading data ---")
    df = load_data(); df = remove_separator_column(df)
    y_all = map_labels(df.iloc[:, 1])
    proteins = df.iloc[:, 0].apply(lambda x: str(x).split('_')[0]).values
    X_250d_df, _ = get_features_and_labels(df)
    pssm_cols = [c for c in X_250d_df.columns if str(c).startswith('pssm') or 'pssm' in str(c).lower()]
    X_109d = X_250d_df.drop(columns=pssm_cols)

    esm3 = np.load(ESM3_79)  # (546, 2560)
    print(f"ESM-3 L79: {esm3.shape}")
    uniprot = np.load(UNIPROT)  # (546, 53)
    print(f"UniProt: {uniprot.shape}")

    # Build feature matrices
    # A: 109D only (baseline)
    # B: 109D + ESM3 = 2669D
    # C: 109D + UniProt = 162D
    # D: 109D + ESM3 + UniProt = 2722D

    X_B = pd.DataFrame(np.hstack([X_109d.values, esm3]), index=X_109d.index)
    X_C = pd.DataFrame(np.hstack([X_109d.values, uniprot]), index=X_109d.index)
    X_D = pd.DataFrame(np.hstack([X_109d.values, esm3, uniprot]), index=X_109d.index)

    print(f"109D: {X_109d.shape}")
    print(f"109D+ESM3: {X_B.shape}")
    print(f"109D+UniProt: {X_C.shape}")
    print(f"109D+ESM3+UniProt: {X_D.shape}")

    # Models: XGB + LGB + Cat
    models_to_use = [(make_xgb, SEED), (make_lgb, SEED), (make_cat, SEED)]
    model_name = "XGB+LGB+Cat"

    results = {}

    # ================================================================
    # Phase 1: 10-fold CV for all configurations
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 1: 10-fold CV ({model_name})")
    print(f"{'='*70}")

    configs = [
        ('109D Baseline', X_109d, None),
        ('109D + UniProt', X_C, None),
        ('109D + ESM-3 L79 (K=1024)', X_B, 1024),
        ('109D + ESM-3 + UniProt (K=1024)', X_D, 1024),
    ]

    for name, X_data, k in configs:
        summary = run_10fold(X_data, y_all, k or 99999, models_to_use)
        print(
            f"  {name:<30} "
            f"AUC={summary['auc']['mean']:.4f}±{summary['auc']['std']:.4f} "
            f"BACC={summary['bacc']['mean']:.4f} MCC={summary['mcc']['mean']:.4f} "
            f"Sn={summary['sn']['mean']:.4f} Sp={summary['sp']['mean']:.4f} "
            f"AP={summary['ap']['mean']:.4f}  [{summary['elapsed_sec']:.1f}s]"
        )
        results[name] = {'10f': summary, **flatten_summary('10f', summary)}

    # ================================================================
    # Phase 2: 10×10 CV for top configurations
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 2: 10×10 CV ({model_name})")
    print(f"{'='*70}")

    for name, X_data, k in configs:
        summary = run_10x10(X_data, y_all, k or 99999, models_to_use)
        print(
            f"  {name:<30} "
            f"10x10 AUC={summary['auc']['mean']:.4f}±{summary['auc']['std']:.4f} "
            f"BACC={summary['bacc']['mean']:.4f} MCC={summary['mcc']['mean']:.4f} "
            f"[{summary['elapsed_sec']:.1f}s]"
        )
        results[name]['10x10'] = summary
        results[name].update(flatten_summary('10x10', summary))

    # Also do single-model baselines for D (best combo)
    print(f"\n  --- Single model baselines for 109D+ESM3+UniProt ---")
    for factory, s, label in [(make_xgb, SEED, 'XGB only'), (make_lgb, SEED, 'LGB only'), (make_cat, SEED, 'Cat only')]:
        summary = run_10x10(X_D, y_all, 1024, [(factory, s)])
        print(f"  {label:<30} 10x10 AUC={summary['auc']['mean']:.4f}±{summary['auc']['std']:.4f}  [{summary['elapsed_sec']:.1f}s]")
        results[f'D_{label}'] = {'10x10': summary, **flatten_summary('10x10', summary)}

    # ================================================================
    # Phase 3: GroupKFold
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 3: GroupKFold ({model_name})")
    print(f"{'='*70}")

    for name, X_data, k in configs:
        summary = run_gkfold(X_data, y_all, proteins, k or 99999, models_to_use)
        print(
            f"  {name:<30} "
            f"GK AUC={summary['auc']['mean']:.4f}±{summary['auc']['std']:.4f} "
            f"BACC={summary['bacc']['mean']:.4f} MCC={summary['mcc']['mean']:.4f} "
            f"[{summary['elapsed_sec']:.1f}s]"
        )
        results[name]['gk'] = summary
        results[name].update(flatten_summary('gk', summary))

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'#'*70}")
    print(f"# FINAL SUMMARY: {model_name} Ensemble")
    print(f"{'#'*70}")
    print(f"{'Configuration':<35} {'10f AUC':>10} {'10x10 AUC':>10} {'GK AUC':>10}")
    print("-" * 70)

    for name, _, _ in configs:
        r = results[name]
        print(f"  {name:<35} {r['10f_auc']:>10.4f} {r['10x10_auc']:>10.4f} {r['gk_auc']:>10.4f}")

    # Deltas
    # Deltas
    bl = results['109D Baseline']
    print(f"\n{'='*70}")
    print("IMPROVEMENTS vs 109D BASELINE:")
    print(f"{'='*70}")
    for name in ['109D + UniProt', '109D + ESM-3 L79 (K=1024)', '109D + ESM-3 + UniProt (K=1024)']:
        r = results[name]
        d_10f = r['10f_auc'] - bl['10f_auc']
        d_10x10 = r['10x10_auc'] - bl['10x10_auc']
        d_gk = r['gk_auc'] - bl['gk_auc']
        print(f"  {name}:")
        print(f"    10-fold: {r['10f_auc']:.4f}  Δ={d_10f:+.4f}")
        print(f"    10x10:   {r['10x10_auc']:.4f}  Δ={d_10x10:+.4f}")
        print(f"    GKfold:  {r['gk_auc']:.4f}  Δ={d_gk:+.4f}")

    # Save to CSV
    csv_path = os.path.join(RESULTS_DIR, 'main_results.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("配置,特征维度,10-fold AUC,10-fold AUC std,10-fold BACC,10-fold MCC,10-fold Sn,10-fold Sp,10x10 AUC,10x10 AUC std,GroupKFold AUC,GroupKFold AUC std\n")
        
        dim_map = {
            '109D Baseline': 109,
            '109D + UniProt': 109 + 53,
            '109D + ESM-3 L79 (K=1024)': 109 + 2560,
            '109D + ESM-3 + UniProt (K=1024)': 109 + 2560 + 53
        }
        
        for name in ['109D Baseline', '109D + UniProt', '109D + ESM-3 L79 (K=1024)', '109D + ESM-3 + UniProt (K=1024)']:
            r = results[name]
            f.write(f"{name},{dim_map[name]},{r['10f_auc']:.4f},{r['10f_auc_std']:.4f},{r['10f_bacc']:.4f},{r['10f_mcc']:.4f},{r['10f_sn']:.4f},{r['10f_sp']:.4f},{r['10x10_auc']:.4f},{r['10x10_auc_std']:.4f},{r['gk_auc']:.4f},{r['gk_auc_std']:.4f}\n")
            
        for label in ['XGB only', 'LGB only', 'Cat only']:
            r = results[f'D_{label}']
            f.write(f"{label} (109D+ESM3+UniProt),2722,—,—,—,—,—,—,{r['10x10_auc']:.4f},{r['10x10_auc_std']:.4f},—,—\n")
            
    print(f"\nSaved to {csv_path}")


if __name__ == '__main__':
    main()
