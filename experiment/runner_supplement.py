"""Supplementary experiments for Analytical Biochemistry submission.

Experiments:
  1. Disease-related UniProt feature ablation
  2. GroupKFold AUPRC (average precision)
  3. GroupKFold AUROC 95% CI (bootstrap)
  4. ANOVA feature-selection K stability (K=256, 512, 1024, 1536)

All experiments use leak-free fold-local feature selection and standardization.
OOF predictions are saved for reproducibility.

Output directory: results/analytical_biochemistry_supplement/
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
from sklearn.utils import resample
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data, remove_separator_column, map_labels, get_features_and_labels

warnings.filterwarnings('ignore')
SEED = 42
BOOTSTRAP_SEED = 2026
BOOTSTRAP_ITERS = 2000

FEAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'features')
ESM3_79 = os.path.join(FEAT_DIR, 'esm3_79', 'X_esm3_79.npy')
UNIPROT = os.path.join(FEAT_DIR, 'X_uniprot.npy')

# Output directory
SUPP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'results', 'analytical_biochemistry_supplement'
)
os.makedirs(SUPP_DIR, exist_ok=True)

# Disease-related UniProt feature indices (alphabetically sorted feature list)
# Index 6: kw_disease, Index 17: n_disease_associations
DISEASE_UNIPROT_IDXS = [6, 17]

# ---------- Model factories (same as main experiment) ----------

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

MODELS = [(make_xgb, SEED), (make_lgb, SEED), (make_cat, SEED)]

# ---------- Core training functions (leak-free) ----------

def select_top_k(X_tr, y_tr, k):
    if k >= X_tr.shape[1]:
        return np.arange(X_tr.shape[1])
    scores, _ = f_classif(X_tr, y_tr)
    return np.argsort(scores)[::-1][:k]

def ensemble_predict_fold(X_tr_raw, X_te_raw, y_tr, k, models_to_use):
    """Train ensemble and return OOF predictions for the test fold."""
    X_tr = X_tr_raw.astype(np.float64)
    X_te = X_te_raw.astype(np.float64)
    y_tr = y_tr.astype(int)

    if k < X_tr.shape[1]:
        idx = select_top_k(X_tr, y_tr, k)
        X_tr = X_tr[:, idx]
        X_te = X_te[:, idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    all_probas = []
    for factory, s in models_to_use:
        m = factory(s)
        m.fit(X_tr_s, y_tr)
        all_probas.append(m.predict_proba(X_te_s)[:, 1])
    return np.mean(all_probas, axis=0)

def binary_metrics(y_true, y_score, threshold=0.5):
    """Compute comprehensive binary classification metrics."""
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

# ---------- CV runners with OOF prediction storage ----------

def run_10fold_with_oof(X, y, k, models_to_use):
    """10-fold CV returning per-fold metrics and pooled OOF predictions."""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_metrics = []
    oof_y_true = np.full(len(y), -1, dtype=float)
    oof_y_score = np.full(len(y), np.nan)
    t0 = time.time()
    for tr, te in skf.split(X, y):
        ys = ensemble_predict_fold(
            X.iloc[tr].values, X.iloc[te].values, y.iloc[tr].values.astype(int),
            k, models_to_use)
        y_te = y.iloc[te].values.astype(int)
        oof_y_true[te] = y_te
        oof_y_score[te] = ys
        fold_metrics.append(binary_metrics(y_te, ys))
    elapsed = time.time() - t0
    summary = summarize_metric_dicts(fold_metrics)
    summary['elapsed_sec'] = elapsed
    # Pooled OOF metrics
    mask = oof_y_true >= 0
    pooled_metrics = binary_metrics(oof_y_true[mask], oof_y_score[mask])
    return summary, fold_metrics, oof_y_true, oof_y_score, pooled_metrics

def run_10x10_with_oof(X, y, k, models_to_use):
    """10x10 CV returning per-fold metrics only (100 folds too many for OOF)."""
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
    return summary, fold_metrics

def run_gkfold_with_oof(X, y, groups, k, models_to_use):
    """GroupKFold returning per-fold metrics and pooled OOF predictions."""
    gkf = GroupKFold(n_splits=10)
    fold_metrics = []
    oof_y_true = np.full(len(y), -1, dtype=float)
    oof_y_score = np.full(len(y), np.nan)
    t0 = time.time()
    for tr, te in gkf.split(X, y, groups):
        ys = ensemble_predict_fold(
            X.iloc[tr].values, X.iloc[te].values, y.iloc[tr].values.astype(int),
            k, models_to_use)
        y_te = y.iloc[te].values.astype(int)
        oof_y_true[te] = y_te
        oof_y_score[te] = ys
        fold_metrics.append(binary_metrics(y_te, ys))
    elapsed = time.time() - t0
    summary = summarize_metric_dicts(fold_metrics)
    summary['elapsed_sec'] = elapsed
    mask = oof_y_true >= 0
    pooled_metrics = binary_metrics(oof_y_true[mask], oof_y_score[mask])
    return summary, fold_metrics, oof_y_true, oof_y_score, pooled_metrics

# ---------- Bootstrap CI ----------

def bootstrap_auroc_ci(y_true, y_score, n_iterations=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED, alpha=0.05):
    """Compute bootstrap 95% CI for AUROC."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)
        # Only bootstrap if both classes are present
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = np.array(aucs)
    lower = np.percentile(aucs, 100 * alpha / 2)
    upper = np.percentile(aucs, 100 * (1 - alpha / 2))
    return {
        'auc_point': roc_auc_score(y_true, y_score),
        'auc_lower_95ci': lower,
        'auc_upper_95ci': upper,
        'bootstrap_seed': seed,
        'bootstrap_iterations': len(aucs),
    }

# ---------- Main ----------

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"\n{'#'*70}")
    print(f"# Supplementary Experiments for Analytical Biochemistry")
    print(f"# {ts}")
    print(f"{'#'*70}")

    # ---- Load Data ----
    print("\n--- Loading data ---")
    df = load_data()
    df = remove_separator_column(df)
    y_all = map_labels(df.iloc[:, 1])
    proteins = df.iloc[:, 0].apply(lambda x: str(x).split('_')[0]).values
    X_250d_df, _ = get_features_and_labels(df)

    # Remove PSSM/WAPSSM → 110D baseline
    pssm_cols = [c for c in X_250d_df.columns if str(c).startswith('pssm') or 'pssm' in str(c).lower()]
    X_110d = X_250d_df.drop(columns=pssm_cols)

    esm3 = np.load(ESM3_79)    # (546, 2560)
    uniprot = np.load(UNIPROT) # (546, 53)

    print(f"110D: {X_110d.shape}, ESM-3: {esm3.shape}, UniProt: {uniprot.shape}")

    # ---- Identify disease-related UniProt features ----
    uniprot_no_disease = np.delete(uniprot, DISEASE_UNIPROT_IDXS, axis=1)
    uniprot_dim_full = uniprot.shape[1]
    uniprot_dim_clean = uniprot_no_disease.shape[1]
    print(f"UniProt features: {uniprot_dim_full} full, {uniprot_dim_clean} without disease-related")

    # ---- Build feature matrices ----
    # A: 110D baseline (dim=110)
    # B: 110D + UniProt (dim=110+53=163)
    # C: 110D + ESM-3 (dim=110+2560=2670)
    # D: 110D + ESM-3 + UniProt full (dim=110+2560+53=2723)
    # E: 110D + ESM-3 + UniProt without disease (dim=110+2560+51=2721)
    # F: 110D + UniProt without disease (dim=110+51=161)

    X_A = X_110d  # 110D baseline, keep as DataFrame
    X_B = pd.DataFrame(np.hstack([X_110d.values, uniprot]), index=X_110d.index)
    X_C = pd.DataFrame(np.hstack([X_110d.values, esm3]), index=X_110d.index)
    X_D = pd.DataFrame(np.hstack([X_110d.values, esm3, uniprot]), index=X_110d.index)
    X_E = pd.DataFrame(np.hstack([X_110d.values, esm3, uniprot_no_disease]), index=X_110d.index)
    X_F = pd.DataFrame(np.hstack([X_110d.values, uniprot_no_disease]), index=X_110d.index)

    # ================================================================
    # Experiment 1: Disease-related UniProt feature ablation
    # ================================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT 1: Disease-related UniProt feature ablation")
    print(f"{'='*70}")

    exp1_configs = [
        ('110D+ESM3+UniProt (full)', X_D, 2723, 1024, 'full'),
        ('110D+ESM3+UniProt (no disease)', X_E, 2721, 1024, 'no_disease'),
    ]

    exp1_results = {}

    for name, X_data, dim, k, tag in exp1_configs:
        print(f"\n  --- {name} (dim={dim}) ---")

        # 10-fold CV with OOF
        summary_10f, folds_10f, oof_y_10f, oof_s_10f, pooled_10f = \
            run_10fold_with_oof(X_data, y_all, k, MODELS)
        print(f"  10-fold:  AUC={summary_10f['auc']['mean']:.4f}±{summary_10f['auc']['std']:.4f}  "
              f"BACC={summary_10f['bacc']['mean']:.4f}  MCC={summary_10f['mcc']['mean']:.4f}  "
              f"AP={summary_10f['ap']['mean']:.4f}  [{summary_10f['elapsed_sec']:.1f}s]")

        # 10x10 CV
        summary_10x10, folds_10x10 = run_10x10_with_oof(X_data, y_all, k, MODELS)
        print(f"  10x10:    AUC={summary_10x10['auc']['mean']:.4f}±{summary_10x10['auc']['std']:.4f}  "
              f"BACC={summary_10x10['bacc']['mean']:.4f}  [{summary_10x10['elapsed_sec']:.1f}s]")

        # GroupKFold with OOF
        summary_gk, folds_gk, oof_y_gk, oof_s_gk, pooled_gk = \
            run_gkfold_with_oof(X_data, y_all, proteins, k, MODELS)
        print(f"  GroupKFold: AUC={summary_gk['auc']['mean']:.4f}±{summary_gk['auc']['std']:.4f}  "
              f"BACC={summary_gk['bacc']['mean']:.4f}  MCC={summary_gk['mcc']['mean']:.4f}  "
              f"AP={summary_gk['ap']['mean']:.4f}  [{summary_gk['elapsed_sec']:.1f}s]")

        exp1_results[tag] = {
            'config': name, 'dim': dim, 'k': k,
            '10f': summary_10f, '10f_folds': folds_10f,
            '10x10': summary_10x10,
            'gk': summary_gk, 'gk_folds': folds_gk,
            'oof_10f_y_true': oof_y_10f.tolist(), 'oof_10f_y_score': oof_s_10f.tolist(),
            'oof_gk_y_true': oof_y_gk.tolist(), 'oof_gk_y_score': oof_s_gk.tolist(),
        }
        # Add flatten
        exp1_results[tag].update(flatten_summary('10f', summary_10f))
        exp1_results[tag].update(flatten_summary('10x10', summary_10x10))
        exp1_results[tag].update(flatten_summary('gk', summary_gk))

    # Save Exp 1 CSVs
    exp1_csv = os.path.join(SUPP_DIR, 'ab_without_disease_uniprot.csv')
    with open(exp1_csv, 'w', encoding='utf-8') as f:
        f.write("配置,特征维度,移除特征,10-fold AUROC,10-fold AUROC std,10-fold BACC,10-fold MCC,"
                "10x10 AUROC,10x10 AUROC std,GroupKFold AUROC,GroupKFold AUROC std,"
                "GroupKFold AP,GroupKFold AP std,random_seed\n")
        for tag in ['full', 'no_disease']:
            r = exp1_results[tag]
            removed = 'none' if tag == 'full' else 'kw_disease, n_disease_associations'
            f.write(f"{r['config']},{r['dim']},{removed},"
                    f"{r['10f_auc']:.4f},{r['10f_auc_std']:.4f},{r['10f_bacc']:.4f},{r['10f_mcc']:.4f},"
                    f"{r['10x10_auc']:.4f},{r['10x10_auc_std']:.4f},"
                    f"{r['gk_auc']:.4f},{r['gk_auc_std']:.4f},"
                    f"{r['gk_ap']:.4f},{r['gk_ap_std']:.4f},"
                    f"{SEED}\n")
    print(f"\nSaved: {exp1_csv}")

    # Save Exp 1 JSON
    exp1_json = exp1_csv.replace('.csv', '.json')
    # Convert numpy arrays for JSON serialization
    exp1_json_safe = {}
    for tag, r in exp1_results.items():
        exp1_json_safe[tag] = {}
        for kk, vv in r.items():
            if isinstance(vv, dict):
                exp1_json_safe[tag][kk] = {k2: v2 for k2, v2 in vv.items()}
            elif isinstance(vv, list) and len(vv) > 0 and isinstance(vv[0], dict):
                exp1_json_safe[tag][kk] = vv  # fold-level metrics
            else:
                exp1_json_safe[tag][kk] = vv
    with open(exp1_json, 'w') as f:
        json.dump(exp1_json_safe, f, indent=2)
    print(f"Saved: {exp1_json}")

    # ================================================================
    # Experiment 2 & 3: GroupKFold AUPRC and AUROC 95% CI
    # (Compute from all 4 main configs' GroupKFold OOF predictions)
    # ================================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT 2 & 3: GroupKFold AUPRC and AUROC 95% CI")
    print(f"{'='*70}")

    exp23_configs = [
        ('110D Baseline', X_A, 110, None),
        ('110D + UniProt', X_B, 163, None),
        ('110D + ESM-3 L79', X_C, 2670, 1024),
        ('110D + ESM-3 + UniProt', X_D, 2723, 1024),
    ]

    exp23_results = {}

    for name, X_data, dim, k in exp23_configs:
        k_eff = k if k else 99999
        print(f"\n  --- {name} (dim={dim}, K={k}) ---")

        summary_gk, folds_gk, oof_y_gk, oof_s_gk, pooled_gk = \
            run_gkfold_with_oof(X_data, y_all, proteins, k_eff, MODELS)

        # Bootstrap CI
        mask = oof_y_gk >= 0
        ci = bootstrap_auroc_ci(oof_y_gk[mask], oof_s_gk[mask])

        print(f"  GroupKFold: AUC={summary_gk['auc']['mean']:.4f}±{summary_gk['auc']['std']:.4f}  "
              f"AP={summary_gk['ap']['mean']:.4f}  AUPRC(pooled)={pooled_gk['ap']:.4f}  "
              f"AUC 95%CI=[{ci['auc_lower_95ci']:.4f}, {ci['auc_upper_95ci']:.4f}]  "
              f"[{summary_gk['elapsed_sec']:.1f}s]")

        exp23_results[name] = {
            'config': name, 'dim': dim, 'k': k_eff if k else None,
            'gk': summary_gk,
            'gk_pooled_auroc': pooled_gk['auc'],
            'gk_pooled_auprc': pooled_gk['ap'],
            'gk_pooled_bacc': pooled_gk['bacc'],
            'gk_pooled_mcc': pooled_gk['mcc'],
            'bootstrap': ci,
            'oof_gk_y_true': oof_y_gk.tolist(),
            'oof_gk_y_score': oof_s_gk.tolist(),
        }
        exp23_results[name].update(flatten_summary('gk', summary_gk))

    # Save Exp 2 (AUPRC) CSV
    exp2_csv = os.path.join(SUPP_DIR, 'ab_groupkfold_auprc.csv')
    with open(exp2_csv, 'w', encoding='utf-8') as f:
        f.write("配置,特征维度,GroupKFold AUROC (fold-mean),GroupKFold AUROC std,"
                "GroupKFold AUROC (pooled OOF),GroupKFold AUPRC (fold-mean),"
                "GroupKFold AUPRC std,GroupKFold AUPRC (pooled OOF),"
                "GroupKFold BACC (pooled),GroupKFold MCC (pooled),"
                "positive_class,random_seed\n")
        for name, _, _, _ in exp23_configs:
            r = exp23_results[name]
            f.write(f"{r['config']},{r['dim']},"
                    f"{r['gk_auc']:.4f},{r['gk_auc_std']:.4f},"
                    f"{r['gk_pooled_auroc']:.4f},"
                    f"{r['gk_ap']:.4f},{r['gk_ap_std']:.4f},"
                    f"{r['gk_pooled_auprc']:.4f},"
                    f"{r['gk_pooled_bacc']:.4f},{r['gk_pooled_mcc']:.4f},"
                    f"pathogenic,{SEED}\n")
    print(f"\nSaved: {exp2_csv}")

    # Save GroupKFold pooled OOF predictions for ROC/PR curve plotting.
    exp23_json = os.path.join(SUPP_DIR, 'ab_groupkfold_oof_predictions.json')
    exp23_json_safe = {}
    for name, r in exp23_results.items():
        exp23_json_safe[name] = {
            'config': r['config'],
            'dim': r['dim'],
            'k': r['k'],
            'gk_pooled_auroc': r['gk_pooled_auroc'],
            'gk_pooled_auprc': r['gk_pooled_auprc'],
            'gk_pooled_bacc': r['gk_pooled_bacc'],
            'gk_pooled_mcc': r['gk_pooled_mcc'],
            'oof_gk_y_true': r['oof_gk_y_true'],
            'oof_gk_y_score': r['oof_gk_y_score'],
        }
    with open(exp23_json, 'w') as f:
        json.dump(exp23_json_safe, f, indent=2)
    print(f"Saved: {exp23_json}")

    # Save Exp 3 (Bootstrap CI) CSV
    exp3_csv = os.path.join(SUPP_DIR, 'ab_groupkfold_auroc_ci.csv')
    with open(exp3_csv, 'w', encoding='utf-8') as f:
        f.write("配置,AUROC (pooled OOF),95% CI lower,95% CI upper,"
                "bootstrap seed,bootstrap iterations,random_seed\n")
        for name, _, _, _ in exp23_configs:
            r = exp23_results[name]
            ci = r['bootstrap']
            f.write(f"{r['config']},{ci['auc_point']:.4f},{ci['auc_lower_95ci']:.4f},"
                    f"{ci['auc_upper_95ci']:.4f},{ci['bootstrap_seed']},"
                    f"{ci['bootstrap_iterations']},{SEED}\n")
    print(f"Saved: {exp3_csv}")

    # ================================================================
    # Experiment 4: ANOVA K value stability
    # ================================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT 4: ANOVA K value stability")
    print(f"{'='*70}")

    K_VALUES = [256, 512, 1024, 1536]

    exp4_results = {}

    for kv in K_VALUES:
        print(f"\n  --- K={kv} ---")
        summary_10x10_k, folds_10x10_k = run_10x10_with_oof(X_D, y_all, kv, MODELS)
        summary_gk_k, folds_gk_k, oof_y_gk_k, oof_s_gk_k, pooled_gk_k = \
            run_gkfold_with_oof(X_D, y_all, proteins, kv, MODELS)

        print(f"  10x10:      AUC={summary_10x10_k['auc']['mean']:.4f}±{summary_10x10_k['auc']['std']:.4f}  "
              f"BACC={summary_10x10_k['bacc']['mean']:.4f}  [{summary_10x10_k['elapsed_sec']:.1f}s]")
        print(f"  GroupKFold: AUC={summary_gk_k['auc']['mean']:.4f}±{summary_gk_k['auc']['std']:.4f}  "
              f"BACC={summary_gk_k['bacc']['mean']:.4f}  [{summary_gk_k['elapsed_sec']:.1f}s]")

        exp4_results[str(kv)] = {
            'k': kv,
            'full_dim': 2723,
            '10x10': summary_10x10_k,
            'gk': summary_gk_k,
        }
        exp4_results[str(kv)].update(flatten_summary('10x10', summary_10x10_k))
        exp4_results[str(kv)].update(flatten_summary('gk', summary_gk_k))

    # Save Exp 4 CSV
    exp4_csv = os.path.join(SUPP_DIR, 'ab_feature_selection_k_stability.csv')
    with open(exp4_csv, 'w', encoding='utf-8') as f:
        f.write("K值,特征维度,10x10 AUROC,10x10 AUROC std,10x10 BACC,10x10 MCC,"
                "GroupKFold AUROC,GroupKFold AUROC std,GroupKFold BACC,GroupKFold MCC,"
                "random_seed\n")
        for kv in K_VALUES:
            r = exp4_results[str(kv)]
            f.write(f"{r['k']},{r['full_dim']},"
                    f"{r['10x10_auc']:.4f},{r['10x10_auc_std']:.4f},"
                    f"{r['10x10_bacc']:.4f},{r['10x10_mcc']:.4f},"
                    f"{r['gk_auc']:.4f},{r['gk_auc_std']:.4f},"
                    f"{r['gk_bacc']:.4f},{r['gk_mcc']:.4f},"
                    f"{SEED}\n")
    print(f"\nSaved: {exp4_csv}")

    # ================================================================
    # Final summary
    # ================================================================
    print(f"\n{'#'*70}")
    print("# SUPPLEMENTARY EXPERIMENTS SUMMARY")
    print(f"{'#'*70}")

    print("\n--- Exp 1: Disease ablation ---")
    for tag in ['full', 'no_disease']:
        r = exp1_results[tag]
        print(f"  {r['config']}: 10f AUC={r['10f_auc']:.4f}  10x10 AUC={r['10x10_auc']:.4f}  "
              f"GK AUC={r['gk_auc']:.4f}  GK AP={r['gk_ap']:.4f}")

    print("\n--- Exp 2&3: GroupKFold AUPRC & 95% CI ---")
    for name, _, _, _ in exp23_configs:
        r = exp23_results[name]
        ci = r['bootstrap']
        print(f"  {name}: GK AUC={r['gk_auc']:.4f}  AP(pooled)={r['gk_pooled_auprc']:.4f}  "
              f"95%CI=[{ci['auc_lower_95ci']:.4f}, {ci['auc_upper_95ci']:.4f}]")

    print("\n--- Exp 4: K stability ---")
    for kv in K_VALUES:
        r = exp4_results[str(kv)]
        print(f"  K={kv}: 10x10 AUC={r['10x10_auc']:.4f}  GK AUC={r['gk_auc']:.4f}  "
              f"GK BACC={r['gk_bacc']:.4f}")

    print(f"\nAll outputs saved to: {SUPP_DIR}/")
    print(f"  {os.path.basename(exp1_csv)}")
    print(f"  {os.path.basename(exp2_csv)}")
    print(f"  {os.path.basename(exp3_csv)}")
    print(f"  {os.path.basename(exp4_csv)}")
    print(f"  {os.path.basename(exp1_json)}")
    print(f"  {os.path.basename(exp23_json)}")


if __name__ == '__main__':
    main()
