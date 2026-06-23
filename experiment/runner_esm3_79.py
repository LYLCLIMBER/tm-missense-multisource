"""Experiment runner for ESM-3 layer 79 features on BorodaTM dataset.

Fast screening with f_classif (ANOVA) for feature selection inside CV,
followed by rigorous 10×10 CV for the best configuration.

Comparison:
  1. Baseline: handcrafted (105D after separator removal)
  2. ESM-3 only (2560D) with feature selection
  3. ESM-3 + Handcrafted fusion (2810D) with feature selection
  4. Previous best: handcrafted (250D from previous experiments)

Author: Claude Opus
"""
import os
import sys
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import run_preprocessing, load_data, remove_separator_column, map_labels

warnings.filterwarnings('ignore')

SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Model ----
def make_xgb():
    return XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.9,
        objective='binary:logistic', random_state=SEED,
        verbosity=0, n_jobs=4, tree_method='hist',
    )

# ---- Metrics ----
def calc_metrics(y_true, y_score):
    y_pred = (y_score >= 0.5).astype(int)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bacc = 0.5 * sn + 0.5 * sp
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return {'AUC': auc, 'AP': ap, 'BACC': bacc, 'MCC': mcc, 'Sn': sn, 'Sp': sp}

def select_top_k(X_train, y_train, k, method='f'):
    """Select top-k features. method='f' for ANOVA (fast), 'mi' for mutual info (slow but better)."""
    if k >= X_train.shape[1]:
        return np.arange(X_train.shape[1])
    if method == 'mi':
        scores = mutual_info_classif(X_train, y_train, random_state=SEED)
    else:
        scores, _ = f_classif(X_train, y_train)
    return np.argsort(scores)[::-1][:k]

# ---- CV functions ----
def single_10fold_cv(X, y, k=None, method='f'):
    """Single 10-fold stratified CV with feature selection inside each fold."""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    all_auc, all_bacc, all_mcc, all_ap = [], [], [], []
    t0 = time.time()
    for tr, te in skf.split(X, y):
        X_tr = X.iloc[tr].values.astype(np.float64)
        X_te = X.iloc[te].values.astype(np.float64)
        y_tr = y.iloc[tr].values.astype(int)
        y_te = y.iloc[te].values.astype(int)
        # Feature selection on training data only
        if k is not None and k < X_tr.shape[1]:
            idx = select_top_k(X_tr, y_tr, k, method)
            X_tr = X_tr[:, idx]; X_te = X_te[:, idx]
        # Standardize
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)
        # Train & predict
        m = make_xgb(); m.fit(X_tr_s, y_tr)
        ys = m.predict_proba(X_te_s)[:, 1]
        metrics = calc_metrics(y_te, ys)
        all_auc.append(metrics['AUC']); all_bacc.append(metrics['BACC'])
        all_mcc.append(metrics['MCC']); all_ap.append(metrics['AP'])
    elapsed = time.time() - t0
    return float(np.mean(all_auc)), float(np.std(all_auc)), float(np.mean(all_bacc)), float(np.mean(all_mcc)), elapsed

def x10_cv(X, y, k=None, method='f'):
    """10×10 CV for rigorous evaluation."""
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=SEED)
    all_auc = []
    t0 = time.time()
    for tr, te in rskf.split(X, y):
        X_tr = X.iloc[tr].values.astype(np.float64)
        X_te = X.iloc[te].values.astype(np.float64)
        y_tr = y.iloc[tr].values.astype(int)
        y_te = y.iloc[te].values.astype(int)
        if k is not None and k < X_tr.shape[1]:
            idx = select_top_k(X_tr, y_tr, k, method)
            X_tr = X_tr[:, idx]; X_te = X_te[:, idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)
        m = make_xgb(); m.fit(X_tr_s, y_tr)
        ys = m.predict_proba(X_te_s)[:, 1]
        all_auc.append(roc_auc_score(y_te, ys))
    elapsed = time.time() - t0
    return float(np.mean(all_auc)), float(np.std(all_auc, ddof=1)), elapsed

def group_kfold_cv(X, y, groups, k=None, method='f'):
    """GroupKFold by protein with feature selection inside each fold."""
    gkf = GroupKFold(n_splits=10)
    all_auc, all_bacc, all_mcc = [], [], []
    t0 = time.time()
    for tr, te in gkf.split(X, y, groups):
        X_tr = X.iloc[tr].values.astype(np.float64)
        X_te = X.iloc[te].values.astype(np.float64)
        y_tr = y.iloc[tr].values.astype(int)
        y_te = y.iloc[te].values.astype(int)
        if k is not None and k < X_tr.shape[1]:
            idx = select_top_k(X_tr, y_tr, k, method)
            X_tr = X_tr[:, idx]; X_te = X_te[:, idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)
        m = make_xgb(); m.fit(X_tr_s, y_tr)
        ys = m.predict_proba(X_te_s)[:, 1]
        metrics = calc_metrics(y_te, ys)
        all_auc.append(metrics['AUC']); all_bacc.append(metrics['BACC']); all_mcc.append(metrics['MCC'])
    elapsed = time.time() - t0
    return float(np.mean(all_auc)), float(np.std(all_auc)), float(np.mean(all_bacc)), float(np.mean(all_mcc)), elapsed

# ---- Main ----
def main():
    print(f"\n{'#'*70}")
    print(f"# ESM-3 Layer 79 Feature Experiments (Fast Screening)")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    # Load data
    print("\n--- Loading data ---")
    df = load_data()
    df = remove_separator_column(df)
    y_all = map_labels(df.iloc[:, 1])
    protein_ids = df.iloc[:, 0].apply(lambda x: str(x).split('_')[0]).values

    # Load feature sets
    from preprocess import get_features_and_labels
    X_250d_df, _ = get_features_and_labels(df)  # Full 250D from Excel (no PLM)
    X_250d = pd.DataFrame(X_250d_df.values, index=X_250d_df.index)
    print(f"[加载] 完整250D特征: {X_250d.shape}")

    X_esm3, _ = run_preprocessing("esm3_only")       # 2560D
    X_esm3_f, _ = run_preprocessing("esm3_fusion")   # 250D + 2560D = 2810D

    # Also create: 250D + ESM3 manually (for MI selection on the combined space)
    X_all_fusion = pd.DataFrame(
        np.hstack([X_250d.values, np.load(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'features', 'esm3_79', 'X_esm3_79.npy'
        ))]),
        index=X_250d.index
    )
    print(f"[加载] 250D+ESM3全融合: {X_all_fusion.shape}")

    results = {}
    METHOD = 'f'  # ANOVA (fast); switch to 'mi' for final validation

    # ================================================================
    # PHASE 1: Quick screening with single 10-fold CV
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 1: Quick Screening (single 10-fold CV, ANOVA feature selection)")
    print(f"{'='*70}")

    # Baselines
    auc, std, bacc, mcc, elapsed = single_10fold_cv(X_250d, y_all)
    print(f"\n  {'Baseline (250D full)':<35} AUC={auc:.4f}±{std:.4f} BACC={bacc:.4f} MCC={mcc:.4f}  [{elapsed:.1f}s]")
    results['baseline_250d'] = {'auc': auc, 'std': std, 'bacc': bacc, 'mcc': mcc}

    # ESM-3 only with different K
    print(f"\n  --- ESM-3 only (2560D) ---")
    for k in [32, 64, 128, 256, 512]:
        auc, std, bacc, mcc, elapsed = single_10fold_cv(X_esm3, y_all, k=k, method=METHOD)
        print(f"  {'esm3_only K='+str(k):<35} AUC={auc:.4f}±{std:.4f} BACC={bacc:.4f} MCC={mcc:.4f}  [{elapsed:.1f}s]")
        results[f'esm3_only_k{k}'] = {'auc': auc, 'std': std, 'bacc': bacc, 'mcc': mcc}

    # ESM-3 + 250D handcrafted (preprocess built-in fusion) with different K
    print(f"\n  --- ESM-3 + 250D Handcrafted fusion (2810D) ---")
    for k in [64, 128, 256, 512, 1024]:
        auc, std, bacc, mcc, elapsed = single_10fold_cv(X_esm3_f, y_all, k=k, method=METHOD)
        print(f"  {'esm3_fusion K='+str(k):<35} AUC={auc:.4f}±{std:.4f} BACC={bacc:.4f} MCC={mcc:.4f}  [{elapsed:.1f}s]")
        results[f'esm3_fusion_k{k}'] = {'auc': auc, 'std': std, 'bacc': bacc, 'mcc': mcc}

    # ================================================================
    # PHASE 2: Rigorous 10×10 CV for best configurations
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: Rigorous 10×10 CV for Top Configurations")
    print(f"{'='*70}")

    # Find best K for each feature set from Phase 1
    best_esm3_k = max([k for k in [32,64,128,256,512]],
                      key=lambda k: results.get(f'esm3_only_k{k}', {}).get('auc', 0))
    best_fusion_k = max([k for k in [64,128,256,512,1024]],
                        key=lambda k: results.get(f'esm3_fusion_k{k}', {}).get('auc', 0))

    print(f"\n  Best K: esm3_only={best_esm3_k}, esm3_fusion={best_fusion_k}")

    # 10×10 CV for top configurations
    auc, std, elapsed = x10_cv(X_250d, y_all)
    print(f"\n  {'Baseline (250D full)':<35} 10x10 AUC={auc:.4f}±{std:.4f}  [{elapsed:.1f}s]")
    results['baseline_250d_10x10'] = {'auc': auc, 'std': std}

    auc, std, elapsed = x10_cv(X_esm3, y_all, k=best_esm3_k, method=METHOD)
    print(f"  {'esm3_only K='+str(best_esm3_k):<35} 10x10 AUC={auc:.4f}±{std:.4f}  [{elapsed:.1f}s]")
    results[f'esm3_only_k{best_esm3_k}_10x10'] = {'auc': auc, 'std': std}

    auc, std, elapsed = x10_cv(X_esm3_f, y_all, k=best_fusion_k, method=METHOD)
    print(f"  {'esm3_fusion K='+str(best_fusion_k):<35} 10x10 AUC={auc:.4f}±{std:.4f}  [{elapsed:.1f}s]")
    results[f'esm3_fusion_k{best_fusion_k}_10x10'] = {'auc': auc, 'std': std}

    # ================================================================
    # PHASE 3: GroupKFold for best configurations
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 3: GroupKFold (Cross-Protein Generalization)")
    print(f"{'='*70}")

    auc, std, bacc, mcc, elapsed = group_kfold_cv(X_250d, y_all, protein_ids)
    print(f"\n  {'Baseline (250D full)':<35} GK AUC={auc:.4f}±{std:.4f} BACC={bacc:.4f} MCC={mcc:.4f}  [{elapsed:.1f}s]")
    results['baseline_250d_gk'] = {'auc': auc, 'std': std, 'bacc': bacc, 'mcc': mcc}

    auc, std, bacc, mcc, elapsed = group_kfold_cv(X_esm3, y_all, protein_ids, k=best_esm3_k, method=METHOD)
    print(f"  {'esm3_only K='+str(best_esm3_k):<35} GK AUC={auc:.4f}±{std:.4f} BACC={bacc:.4f} MCC={mcc:.4f}  [{elapsed:.1f}s]")
    results[f'esm3_only_k{best_esm3_k}_gk'] = {'auc': auc, 'std': std, 'bacc': bacc, 'mcc': mcc}

    auc, std, bacc, mcc, elapsed = group_kfold_cv(X_esm3_f, y_all, protein_ids, k=best_fusion_k, method=METHOD)
    print(f"  {'esm3_fusion K='+str(best_fusion_k):<35} GK AUC={auc:.4f}±{std:.4f} BACC={bacc:.4f} MCC={mcc:.4f}  [{elapsed:.1f}s]")
    results[f'esm3_fusion_k{best_fusion_k}_gk'] = {'auc': auc, 'std': std, 'bacc': bacc, 'mcc': mcc}

    # ================================================================
    # Save results
    # ================================================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(RESULTS_DIR, f'esm3_79_{timestamp}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # ================================================================
    # Final Summary
    # ================================================================
    print(f"\n{'#'*70}")
    print(f"# FINAL SUMMARY")
    print(f"{'#'*70}")
    print(f"{'Experiment':<40} {'AUC':>8} {'Std':>8} {'Protocol':>15}")
    print("-" * 75)
    summary_items = [
        ('baseline_250d', 'Baseline (250D full)', '10-fold CV'),
        ('baseline_105d', 'Baseline (105D hc only)', '10-fold CV'),
        (f'esm3_only_k{best_esm3_k}', f'ESM-3 only K={best_esm3_k}', '10-fold CV'),
        (f'esm3_fusion_k{best_fusion_k}', f'ESM-3+250D Fusion K={best_fusion_k}', '10-fold CV'),
        ('baseline_250d_10x10', 'Baseline (250D full)', '10x10 CV'),
        (f'esm3_only_k{best_esm3_k}_10x10', f'ESM-3 only K={best_esm3_k}', '10x10 CV'),
        (f'esm3_fusion_k{best_fusion_k}_10x10', f'ESM-3+250D Fusion K={best_fusion_k}', '10x10 CV'),
        ('baseline_250d_gk', 'Baseline (250D full)', 'GroupKFold'),
        (f'esm3_only_k{best_esm3_k}_gk', f'ESM-3 only K={best_esm3_k}', 'GroupKFold'),
        (f'esm3_fusion_k{best_fusion_k}_gk', f'ESM-3+250D Fusion K={best_fusion_k}', 'GroupKFold'),
    ]
    for key, name, protocol in summary_items:
        if key in results:
            r = results[key]
            print(f"{name:<40} {r['auc']:>8.4f} {r['std']:>8.4f} {protocol:>15}")

    # Key comparisons
    print(f"\n{'='*70}")
    print("KEY COMPARISONS vs 250D BASELINE:")
    print(f"{'='*70}")
    bl = results.get('baseline_250d', {}).get('auc', 0)
    bl_10x10 = results.get('baseline_250d_10x10', {}).get('auc', 0)
    bl_gk = results.get('baseline_250d_gk', {}).get('auc', 0)
    esm3 = results.get(f'esm3_only_k{best_esm3_k}', {}).get('auc', 0)
    esm3_10x10 = results.get(f'esm3_only_k{best_esm3_k}_10x10', {}).get('auc', 0)
    esm3_gk = results.get(f'esm3_only_k{best_esm3_k}_gk', {}).get('auc', 0)
    fus = results.get(f'esm3_fusion_k{best_fusion_k}', {}).get('auc', 0)
    fus_10x10 = results.get(f'esm3_fusion_k{best_fusion_k}_10x10', {}).get('auc', 0)
    fus_gk = results.get(f'esm3_fusion_k{best_fusion_k}_gk', {}).get('auc', 0)
    print(f"  10-fold CV:  250D={bl:.4f}  ESM3-only={esm3:.4f} ({esm3-bl:+.4f})  ESM3+Fusion={fus:.4f} ({fus-bl:+.4f})")
    print(f"  10x10 CV:    250D={bl_10x10:.4f}  ESM3-only={esm3_10x10:.4f} ({esm3_10x10-bl_10x10:+.4f})  ESM3+Fusion={fus_10x10:.4f} ({fus_10x10-bl_10x10:+.4f})")
    print(f"  GroupKFold:  250D={bl_gk:.4f}  ESM3-only={esm3_gk:.4f} ({esm3_gk-bl_gk:+.4f})  ESM3+Fusion={fus_gk:.4f} ({fus_gk-bl_gk:+.4f})")

    print(f"\nResults saved to: {out_path}")

if __name__ == '__main__':
    main()
