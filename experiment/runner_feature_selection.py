"""Refined feature selection: K-scan + RFE + significance-based + Optuna.

Part A: Feature selection method comparison
  - Finer K-scan (128 to 2560)
  - RFE ranking with fast estimator
  - ANOVA significance threshold (p < 0.05)
  - All inside 10-fold CV (leak-free)

Part B: Optuna hyperparameter tuning (nested CV)
  - Use best feature selection from Part A
"""
import numpy as np, pandas as pd, time, sys
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
sys.path.insert(0, '.')
from preprocess import load_data, remove_separator_column, map_labels, get_features_and_labels
import warnings; warnings.filterwarnings('ignore')

SEED = 42

# Load
df = load_data(); df = remove_separator_column(df)
y_all = map_labels(df.iloc[:, 1]).values.astype(int)
proteins = df.iloc[:, 0].apply(lambda x: str(x).split('_')[0]).values
X_250d_df, _ = get_features_and_labels(df)
X_250d = X_250d_df.values.astype(np.float64)
esm3 = np.load('features/esm3_79/X_esm3_79.npy').astype(np.float64)
uniprot = np.load('features/X_uniprot.npy').astype(np.float64)
X_all = np.hstack([X_250d, esm3, uniprot])  # (546, 2863)
D = X_all.shape[1]
print(f'X shape: {X_all.shape}')

def make_xgb(s=SEED):
    return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.9,
                         objective='binary:logistic', random_state=s,
                         verbosity=0, n_jobs=4, tree_method='hist')

# ================================================================
# Part A: Feature Selection Methods
# ================================================================
print('\n' + '='*60)
print('PART A: Feature Selection Method Comparison')
print('='*60)

# A1: One-time full-dataset analysis (for understanding, NOT for final eval)
print('\n--- A1: Full-dataset feature analysis (guidance only) ---')
# ANOVA p-values
scores, pvals = f_classif(X_all, y_all)
n_sig = np.sum(pvals < 0.05)
n_sig_bonf = np.sum(pvals < 0.05 / D)  # Bonferroni correction
print(f'  ANOVA p<0.05: {n_sig}/{D} features significant')
print(f'  ANOVA p<{0.05/D:.6f} (Bonferroni): {n_sig_bonf}/{D} features')

# RFE ranking (fast: LogisticRegression, step=50)
print('  Running RFE (LogisticRegression, step=50)...')
t0 = time.time()
rfe = RFE(LogisticRegression(max_iter=2000, random_state=SEED),
          n_features_to_select=1, step=50)
rfe.fit(StandardScaler().fit_transform(X_all), y_all)
rfe_ranking = rfe.ranking_  # 1=best, higher=worse
print(f'  RFE done [{time.time()-t0:.1f}s]')

# Get feature rankings for use inside CV
anova_ranking = np.argsort(scores)[::-1]  # high score = important
rfe_order = np.argsort(rfe_ranking)       # low ranking = important
# How much do ANOVA and RFE agree?
overlap_top_k = []
for k in [128, 256, 512, 1024, 2048]:
    anova_set = set(anova_ranking[:k])
    rfe_set = set(rfe_order[:k])
    overlap_top_k.append((k, len(anova_set & rfe_set) / k * 100))
print(f'  ANOVA vs RFE overlap: {dict(overlap_top_k)}')

# A2: 10-fold CV comparing methods
print('\n--- A2: 10-fold CV comparison ---')
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

def select_features(X_tr, y_tr, k, method, ranking=None):
    """Select features. method: 'anova_k', 'anova_sig', 'rfe_k'"""
    if method == 'anova_k':
        scores, _ = f_classif(X_tr, y_tr)
        return np.argsort(scores)[::-1][:k]
    elif method == 'anova_sig':
        _, pvals = f_classif(X_tr, y_tr)
        # Bonferroni-corrected significance
        threshold = 0.05 / X_tr.shape[1]
        return np.where(pvals < threshold)[0]
    elif method == 'rfe_k':
        # Use pre-computed RFE ranking (has leakage from full data - for comparison only)
        # Take top-k from the pre-computed RFE order
        return rfe_order[:k]
    else:  # 'none'
        return np.arange(X_tr.shape[1])

def eval_10fold(X, y, k, method, ranking=None):
    aucs = []; t0 = time.time()
    for tr, te in skf.split(X, y):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        idx = select_features(X_tr, y_tr, k, method, ranking)
        if len(idx) == 0:
            idx = np.arange(min(10, X_tr.shape[1]))
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr[:, idx])
        X_te_s = scaler.transform(X_te[:, idx])
        m = make_xgb(); m.fit(X_tr_s, y_tr)
        aucs.append(roc_auc_score(y_te, m.predict_proba(X_te_s)[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs)), time.time()-t0

results = {}

# ANOVA K-scan (finer)
print('\n  ANOVA K-scan:')
for k in [64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 2560]:
    auc, std, elapsed = eval_10fold(X_all, y_all, k, 'anova_k')
    print(f'    K={k:<5}  AUC={auc:.4f} +/- {std:.4f}  [{elapsed:.1f}s]')
    results[f'anova_k{k}'] = {'auc': auc, 'std': std}

# ANOVA significance
auc, std, elapsed = eval_10fold(X_all, y_all, 0, 'anova_sig')
n_selected = 'varies'
print(f'    ANOVA sig (Bonferroni): AUC={auc:.4f} +/- {std:.4f}  [{elapsed:.1f}s]')
results['anova_sig'] = {'auc': auc, 'std': std}

# RFE K-scan (using pre-computed ranking - has leakage, upper bound only)
print('\n  RFE K-scan (pre-computed ranking, upper bound):')
for k in [256, 512, 1024, 1536, 2048]:
    auc, std, elapsed = eval_10fold(X_all, y_all, k, 'rfe_k')
    print(f'    K={k:<5}  AUC={auc:.4f} +/- {std:.4f}  [{elapsed:.1f}s]')
    results[f'rfe_k{k}'] = {'auc': auc, 'std': std}

# No selection baseline
auc, std, elapsed = eval_10fold(X_all, y_all, 0, 'none')
print(f'\n  No selection (2863D): AUC={auc:.4f} +/- {std:.4f}  [{elapsed:.1f}s]')
results['none'] = {'auc': auc, 'std': std}

# Find best
best_anova = max([(k, results[f'anova_k{k}']['auc']) for k in [64,128,256,384,512,768,1024,1536,2048,2560]], key=lambda x: x[1])
print(f'\n  Best ANOVA: K={best_anova[0]} AUC={best_anova[1]:.4f}')

# ================================================================
# Part B: 10x10 CV for top feature selection methods
# ================================================================
print('\n' + '='*60)
print('PART B: 10x10 CV for Top Methods')
print('='*60)

best_k = best_anova[0]
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=SEED)

for name, method, k in [('ANOVA K='+str(best_k), 'anova_k', best_k),
                          ('ANOVA K=1024 (ref)', 'anova_k', 1024),
                          ('ANOVA sig', 'anova_sig', 0)]:
    aucs = []; t0 = time.time()
    for tr, te in rskf.split(X_all, y_all):
        idx = select_features(X_all[tr], y_all[tr], k, method)
        if len(idx) == 0:
            idx = np.arange(min(10, X_all.shape[1]))
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_all[tr][:, idx])
        X_te_s = scaler.transform(X_all[te][:, idx])
        m = make_xgb(); m.fit(X_tr_s, y_all[tr])
        aucs.append(roc_auc_score(y_all[te], m.predict_proba(X_te_s)[:, 1]))
    auc = float(np.mean(aucs)); std = float(np.std(aucs, ddof=1))
    print(f'  {name:<25} 10x10 AUC={auc:.4f} +/- {std:.4f}  [{time.time()-t0:.1f}s]')
    results[f'{name}_10x10'] = {'auc': auc, 'std': std}

# ================================================================
# Summary
# ================================================================
print('\n' + '='*60)
print('FEATURE SELECTION SUMMARY')
print('='*60)
print(f'{"Method":<25} {"10-fold AUC":>12} {"10x10 AUC":>12}')
print('-' * 50)
bl_auc = results.get('anova_k1024', {}).get('auc', 0)
for name, key_10f, key_10x10 in [
    ('No selection', 'none', None),
    ('ANOVA K=1024 (ref)', 'anova_k1024', 'ANOVA K=1024 (ref)_10x10'),
    (f'ANOVA K={best_k} (best)', f'anova_k{best_k}', f'ANOVA K={best_k}_10x10'),
    ('ANOVA sig (Bonferroni)', 'anova_sig', 'ANOVA sig_10x10'),
]:
    v10f = results.get(key_10f, {}).get('auc', 0)
    v10x10 = results.get(key_10x10, {}).get('auc', 0) if key_10x10 else '—'
    print(f'  {name:<25} {v10f:>12.4f} {str(v10x10):>12}')

print(f'\nBest feature selection: ANOVA K={best_k}')
print('Ready for Optuna tuning in Part C.')
