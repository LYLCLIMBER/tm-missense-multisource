"""Benchmark external tools (SIFT, PolyPhen-2, PROVEAN, FATHMM) on BorodaTM 546.

Uses tool prediction scores already present in data.xlsx. Evaluates under the same
3 CV protocols (10-fold, 10x10, GroupKFold) as our model, plus ESM-1v zero-shot.
"""

import os, sys, json, time, warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data, remove_separator_column, map_labels

warnings.filterwarnings('ignore')
SEED = 42


def load_tool_scores(df):
    """Extract tool prediction columns from the dataframe."""
    scores = {}
    # SIFT_SCORE: lower = more damaging. Threshold < 0.05 = damaging.
    if 'SIFT_SCORE' in df.columns:
        s = df['SIFT_SCORE'].values.astype(float)
        # NaN handling
        s = np.nan_to_num(s, nan=np.nanmedian(s))
        scores['SIFT'] = s

    # pph2_prob (PolyPhen-2): higher = more damaging. Threshold > 0.5 = damaging.
    if 'pph2_prob' in df.columns:
        s = df['pph2_prob'].values.astype(float)
        s = np.nan_to_num(s, nan=np.nanmedian(s))
        scores['PolyPhen-2'] = s

    # PROVEAN_SCORE: lower = more damaging. Threshold <= -2.5 = damaging.
    if 'PROVEAN_SCORE' in df.columns:
        s = df['PROVEAN_SCORE'].values.astype(float)
        s = np.nan_to_num(s, nan=np.nanmedian(s))
        scores['PROVEAN'] = s

    # fathmm_Score: lower = more damaging. Threshold <= -1.5 = damaging.
    if 'fathmm_Score' in df.columns:
        s = df['fathmm_Score'].values.astype(float)
        s = np.nan_to_num(s, nan=np.nanmedian(s))
        scores['FATHMM'] = s

    return scores


def load_esm1v_zeroshot():
    """Load pre-computed ESM-1v zero-shot scores."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'improved_results', 'esm_zeroshot_scores.npy')
    if os.path.exists(path):
        return np.load(path)
    return None


def binary_metrics(y_true, y_score, threshold=0.5):
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
        'sn': sn, 'sp': sp,
        'mcc': matthews_corrcoef(y_true, y_pred),
    }


def summarize(metric_dicts):
    keys = metric_dicts[0].keys()
    return {k: {'mean': float(np.mean([m[k] for m in metric_dicts])),
                'std': float(np.std([m[k] for m in metric_dicts], ddof=1))
                } for k in keys}


def make_binary_predictions(score, y_true, tool_name):
    """For tools where lower score = damaging, negate to make higher = pathogenic.

    Standard thresholds:
      SIFT: < 0.05 = damaging
      PolyPhen-2: > 0.5 = damaging (already correct direction)
      PROVEAN: <= -2.5 = damaging
      FATHMM: <= -1.5 = damaging
    """
    if tool_name == 'SIFT':
        # score < 0.05 -> pathogenic (1)
        return (score < 0.05).astype(int)
    elif tool_name == 'PolyPhen-2':
        # score > 0.5 -> pathogenic (1)
        return (score > 0.5).astype(int)
    elif tool_name == 'PROVEAN':
        return (score <= -2.5).astype(int)
    elif tool_name == 'FATHMM':
        return (score <= -1.5).astype(int)
    elif tool_name == 'ESM-1v-zero-shot':
        # score < 0 -> pathogenic (1)
        return (score < 0).astype(int)
    else:
        return (score >= 0.5).astype(int)


def get_auc_score(score, y_true, tool_name):
    """Get properly directed AUROC.

    For tools where lower score = pathogenic, negate so that higher = pathogenic.
    """
    if tool_name in ('SIFT', 'PROVEAN', 'FATHMM', 'ESM-1v-zero-shot'):
        return roc_auc_score(y_true, -score)
    else:
        return roc_auc_score(y_true, score)


def evaluate_tool_cv(score_array, y_all, proteins, tool_name):
    """Evaluate a single tool under all 3 CV protocols."""
    N = len(y_all)
    y = y_all.values.astype(int) if hasattr(y_all, 'values') else np.asarray(y_all).astype(int)

    results = {}

    # --- 10-fold stratified CV ---
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_metrics = []
    t0 = time.time()
    for tr, te in skf.split(np.zeros(N), y):
        y_te = y[te]
        y_score = score_array[te]
        fold_metrics.append(binary_metrics(
            y_te, y_score,
            threshold=0.5 if tool_name == 'PolyPhen-2' else 0
        ))
        # Actually for threshold-based metrics, we use the standard thresholds
        # Override: use make_binary_predictions for BACC/MCC etc.
    elapsed = time.time() - t0

    # Re-do with proper AUC and threshold
    cv_aucs, cv_baccs, cv_mccs, cv_aps = [], [], [], []
    for tr, te in skf.split(np.zeros(N), y):
        y_te = y[te]
        s = score_array[te]
        cv_aucs.append(get_auc_score(s, y_te, tool_name))
        cv_aps.append(average_precision_score(
            y_te, -s if tool_name in ('SIFT', 'PROVEAN', 'FATHMM', 'ESM-1v-zero-shot') else s))
        y_pred = make_binary_predictions(s, y_te, tool_name)
        cv_baccs.append(balanced_accuracy_score(y_te, y_pred))
        cv_mccs.append(matthews_corrcoef(y_te, y_pred))

    results['10f'] = {
        'auc': {'mean': float(np.mean(cv_aucs)), 'std': float(np.std(cv_aucs))},
        'ap': {'mean': float(np.mean(cv_aps)), 'std': float(np.std(cv_aps))},
        'bacc': {'mean': float(np.mean(cv_baccs)), 'std': float(np.std(cv_baccs))},
        'mcc': {'mean': float(np.mean(cv_mccs)), 'std': float(np.std(cv_mccs))},
        'elapsed_sec': elapsed,
    }

    # --- 10x10 CV ---
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=SEED)
    aucs_10x10 = []
    t0 = time.time()
    for tr, te in rskf.split(np.zeros(N), y):
        aucs_10x10.append(get_auc_score(score_array[te], y[te], tool_name))
    elapsed = time.time() - t0
    results['10x10'] = {
        'auc': {'mean': float(np.mean(aucs_10x10)), 'std': float(np.std(aucs_10x10, ddof=1))},
        'elapsed_sec': elapsed,
    }

    # --- GroupKFold ---
    gkf = GroupKFold(n_splits=10)
    gk_aucs, gk_baccs, gk_mccs = [], [], []
    t0 = time.time()
    for tr, te in gkf.split(np.zeros(N), y, proteins):
        y_te = y[te]
        s = score_array[te]
        gk_aucs.append(get_auc_score(s, y_te, tool_name))
        y_pred = make_binary_predictions(s, y_te, tool_name)
        gk_baccs.append(balanced_accuracy_score(y_te, y_pred))
        gk_mccs.append(matthews_corrcoef(y_te, y_pred))
    elapsed = time.time() - t0
    results['gk'] = {
        'auc': {'mean': float(np.mean(gk_aucs)), 'std': float(np.std(gk_aucs))},
        'bacc': {'mean': float(np.mean(gk_baccs)), 'std': float(np.std(gk_baccs))},
        'mcc': {'mean': float(np.mean(gk_mccs)), 'std': float(np.std(gk_mccs))},
        'elapsed_sec': elapsed,
    }

    return results


def main():
    print(f"{'#'*70}")
    print(f"# External Tool Benchmark on BorodaTM 546")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    # Load data
    print("Loading data...")
    df = load_data()
    df = remove_separator_column(df)
    y_all = map_labels(df.iloc[:, 1])  # 1=pathogenic, 0=benign
    proteins = df.iloc[:, 0].apply(lambda x: str(x).split('_')[0]).values
    print(f"  Samples: {len(y_all)}, Pathogenic: {y_all.sum()}, Benign: {(1-y_all).sum()}")
    print(f"  Proteins: {len(set(proteins))}")

    # Load tool scores
    tool_scores = load_tool_scores(df)
    print(f"\nLoaded {len(tool_scores)} tool scores:")
    for name, scores in tool_scores.items():
        print(f"  {name}: {len(scores)} scores, "
              f"min={scores.min():.4f}, max={scores.max():.4f}, "
              f"nan={np.isnan(scores).sum()}")

    # Load ESM-1v zero-shot
    esm_zs = load_esm1v_zeroshot()
    if esm_zs is not None:
        nonzero = (esm_zs != 0).sum()
        print(f"  ESM-1v-zero-shot: {len(esm_zs)} scores, "
              f"min={esm_zs.min():.4f}, max={esm_zs.max():.4f}, "
              f"nonzero={nonzero}/{len(esm_zs)}")
        tool_scores['ESM-1v-zero-shot'] = esm_zs

    # Evaluate each tool
    all_results = {}
    for tool_name, scores in tool_scores.items():
        print(f"\n{'='*70}")
        print(f"Evaluating: {tool_name}")
        print(f"{'='*70}")
        results = evaluate_tool_cv(scores, y_all, proteins, tool_name)
        all_results[tool_name] = results

        print(f"  10-fold  AUC={results['10f']['auc']['mean']:.4f}±{results['10f']['auc']['std']:.4f}  "
              f"BACC={results['10f']['bacc']['mean']:.4f}  MCC={results['10f']['mcc']['mean']:.4f}  "
              f"[{results['10f']['elapsed_sec']:.1f}s]")
        print(f"  10×10    AUC={results['10x10']['auc']['mean']:.4f}±{results['10x10']['auc']['std']:.4f}  "
              f"[{results['10x10']['elapsed_sec']:.1f}s]")
        print(f"  GroupKF  AUC={results['gk']['auc']['mean']:.4f}±{results['gk']['auc']['std']:.4f}  "
              f"BACC={results['gk']['bacc']['mean']:.4f}  MCC={results['gk']['mcc']['mean']:.4f}  "
              f"[{results['gk']['elapsed_sec']:.1f}s]")

    # ---- Load our model results for comparison ----
    print(f"\n{'='*70}")
    print("Loading our model results for comparison...")
    print(f"{'='*70}")

    our_results = {}
    result_files = [
        ('Our 250D Baseline', 'results/final_ensemble_20260531_151056.json', '250D Baseline'),
        ('Our 250D+UniProt', 'results/final_ensemble_20260531_151056.json', '250D + UniProt'),
        ('Our 250D+ESM3+UniProt', 'results/final_ensemble_20260531_151056.json', '250D + ESM3 + UniProt'),
    ]

    for label, fname, key in result_files:
        fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            if key in data:
                r = data[key]
                our_results[label] = {
                    '10f_auc': r.get('10f_auc', float('nan')),
                    '10f_bacc': r.get('10f_bacc', float('nan')),
                    '10f_mcc': r.get('10f_mcc', float('nan')),
                    '10x10_auc': r.get('10x10_auc', float('nan')),
                    'gk_auc': r.get('gk_auc', float('nan')),
                    'gk_bacc': r.get('gk_bacc', float('nan')),
                    'gk_mcc': r.get('gk_mcc', float('nan')),
                }
                print(f"  {label}: 10f AUC={our_results[label]['10f_auc']:.4f}, "
                      f"10x10 AUC={our_results[label]['10x10_auc']:.4f}, "
                      f"GK AUC={our_results[label]['gk_auc']:.4f}")

    # ---- Summary Table ----
    print(f"\n{'#'*70}")
    print(f"# FINAL COMPARISON TABLE")
    print(f"{'#'*70}")

    print(f"\n{'Method':<30} {'10-fold AUC':>12} {'10×10 AUC':>12} {'GK AUC':>12} {'BACC':>8} {'MCC':>8}")
    print("-" * 78)

    # Our models
    for label, r in our_results.items():
        print(f"  {label:<28} {r['10f_auc']:>12.4f} {r['10x10_auc']:>12.4f} {r['gk_auc']:>12.4f} "
              f"{r['10f_bacc']:>8.4f} {r['10f_mcc']:>8.4f}")

    print("-" * 78)

    # External tools
    for tool_name in ['SIFT', 'PolyPhen-2', 'PROVEAN', 'FATHMM', 'ESM-1v-zero-shot']:
        if tool_name in all_results:
            r = all_results[tool_name]
            print(f"  {tool_name:<28} {r['10f']['auc']['mean']:>12.4f} {r['10x10']['auc']['mean']:>12.4f} "
                  f"{r['gk']['auc']['mean']:>12.4f} "
                  f"{r['10f']['bacc']['mean']:>8.4f} {r['10f']['mcc']['mean']:>8.4f}")

    # ---- Save ----
    # Flatten for JSON serialization
    flat_results = {}
    for name, r in all_results.items():
        flat_results[name] = {
            '10f_auc': r['10f']['auc']['mean'], '10f_auc_std': r['10f']['auc']['std'],
            '10f_bacc': r['10f']['bacc']['mean'], '10f_bacc_std': r['10f']['bacc']['std'],
            '10f_mcc': r['10f']['mcc']['mean'], '10f_mcc_std': r['10f']['mcc']['std'],
            '10f_ap': r['10f']['ap']['mean'], '10f_ap_std': r['10f']['ap']['std'],
            '10x10_auc': r['10x10']['auc']['mean'], '10x10_auc_std': r['10x10']['auc']['std'],
            'gk_auc': r['gk']['auc']['mean'], 'gk_auc_std': r['gk']['auc']['std'],
            'gk_bacc': r['gk']['bacc']['mean'], 'gk_bacc_std': r['gk']['bacc']['std'],
            'gk_mcc': r['gk']['mcc']['mean'], 'gk_mcc_std': r['gk']['mcc']['std'],
        }
    flat_results['our_models'] = our_results

    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(RESULTS_DIR, f'tool_comparison_{timestamp}.json')
    with open(out_path, 'w') as f:
        json.dump(flat_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
