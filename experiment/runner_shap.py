"""SHAP feature importance analysis for the best model configuration.

Analyzes: 250D + ESM-3 L79 + UniProt features, XGBoost, K=1024.
Breaks down importance by feature source (handcrafted, ESM-3, UniProt).
"""

import os, sys, json, time, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

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

K = 1024

print("Loading data...")
df = load_data(); df = remove_separator_column(df)
y_all = map_labels(df.iloc[:, 1]).values.astype(int)
proteins = df.iloc[:, 0].apply(lambda x: str(x).split('_')[0]).values
X_250d_df, feat_names_250d = get_features_and_labels(df)
X_250d = X_250d_df.values.astype(np.float64)

esm3 = np.load(ESM3_79).astype(np.float64)
uniprot = np.load(UNIPROT).astype(np.float64)

# Feature names - use original DataFrame column names
# After removing separator, the feature columns are: df.iloc[:, 2:] excluding the separator
df_feat = df.iloc[:, 2:]
hc_names = [str(c) for c in df_feat.columns]
print(f"hc_names count: {len(hc_names)}, sample: {hc_names[:5]}...")
esm3_names = [f'ESM3_{i}' for i in range(esm3.shape[1])]
up_names = [f'UP_{i}' for i in range(uniprot.shape[1])]
all_feat_names = hc_names + esm3_names + up_names

X_all = np.hstack([X_250d, esm3, uniprot])
print(f"Full feature matrix: {X_all.shape}  |  K={K}")
print(f"  Handcrafted: {len(hc_names)}")
print(f"  ESM-3 L79:   {len(esm3_names)}")
print(f"  UniProt:     {len(up_names)}")

# ---- Feature selection on full data (for interpretation only) ----
scores, _ = f_classif(X_all, y_all)
top_idx = np.argsort(scores)[::-1][:K]
X_sel = X_all[:, top_idx]
sel_names = [all_feat_names[i] for i in top_idx]
print(f"Selected {K} features via ANOVA")

# ---- Train XGBoost on full data ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel)

model = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.9,
    objective='binary:logistic', random_state=SEED,
    verbosity=0, n_jobs=4, tree_method='hist',
)
model.fit(X_scaled, y_all)
print(f"Trained XGBoost, train AUC={roc_auc_score(y_all, model.predict_proba(X_scaled)[:,1]):.4f}")

# ---- SHAP Analysis ----
print("\nComputing SHAP values...")
import shap
t0 = time.time()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)
print(f"  SHAP computed [{time.time()-t0:.1f}s], shape={shap_values.shape}")

# Mean |SHAP| per feature
mean_shap = np.abs(shap_values).mean(axis=0)
shap_order = np.argsort(mean_shap)[::-1]

# ---- Categorize features ----
# Determine source category for each selected feature
hc_count = len(hc_names)
esm3_count = len(esm3_names)
up_count = len(up_names)

source_labels = []
source_indices = {'Handcrafted': [], 'ESM-3': [], 'UniProt': []}
for rank_i, feat_idx in enumerate(top_idx):
    if feat_idx < hc_count:
        source_labels.append('Handcrafted')
        source_indices['Handcrafted'].append(rank_i)
    elif feat_idx < hc_count + esm3_count:
        source_labels.append('ESM-3')
        source_indices['ESM-3'].append(rank_i)
    else:
        source_labels.append('UniProt')
        source_indices['UniProt'].append(rank_i)

# ---- Output ----
print(f"\n{'='*70}")
print("SHAP Feature Importance Analysis")
print(f"{'='*70}")

# Top 30 features
print(f"\n{'Rank':<6} {'Feature':<40} {'Source':<15} {'Mean|SHAP|':>12}")
print("-" * 78)
for r, feat_idx in enumerate(shap_order[:30]):
    name = sel_names[feat_idx]
    if len(name) > 38:
        name = name[:35] + '...'
    print(f"  {r+1:<4} {name:<40} {source_labels[feat_idx]:<15} {mean_shap[feat_idx]:>12.6f}")

# Source-level summary
print(f"\n{'='*70}")
print("IMPORTANCE BY FEATURE SOURCE")
print(f"{'='*70}")
print(f"{'Source':<20} {'N Selected':>12} {'Total|SHAP|':>15} {'Share %':>10} {'Mean|SHAP|':>12}")
print("-" * 72)
total_shap = mean_shap.sum()
for source in ['Handcrafted', 'ESM-3', 'UniProt']:
    idx = source_indices[source]
    n_sel = len(idx)
    sum_shap = mean_shap[idx].sum()
    mean_per_feat = mean_shap[idx].mean()
    print(f"  {source:<18} {n_sel:>12} {sum_shap:>15.4f} {100*sum_shap/total_shap:>9.1f}% {mean_per_feat:>12.6f}")

print("-" * 72)
print(f"  {'TOTAL':<18} {K:>12} {total_shap:>15.4f} {100:>9.1f}%")

# Top features by source
for source in ['Handcrafted', 'ESM-3', 'UniProt']:
    idx = source_indices[source]
    if len(idx) == 0:
        continue
    source_order = [i for i in shap_order if i in idx]
    print(f"\n--- Top {source} features ---")
    for r, feat_idx in enumerate(source_order[:10]):
        name = sel_names[feat_idx]
        if len(name) > 50:
            name = name[:47] + '...'
        print(f"  {r+1}. {name}  SHAP={mean_shap[feat_idx]:.6f}")

# ---- Plots ----
print(f"\n{'='*70}")
print("GENERATING PLOTS")
print(f"{'='*70}")

# Use Chinese-compatible font
plt.rcParams['font.family'] = 'sans-serif'
try:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
except:
    pass

# 1. Top 30 SHAP bar plot
fig, ax = plt.subplots(figsize=(10, 10))
top30 = shap_order[:30]
names_30 = [sel_names[i] for i in top30]
# Shorten long names
names_30_short = []
for n in names_30:
    if len(n) > 45:
        n = n[:42] + '...'
    names_30_short.append(n)

y_pos = range(len(top30))
ax.barh(y_pos, mean_shap[top30][::-1], color='steelblue')
ax.set_yticks(y_pos)
ax.set_yticklabels(names_30_short[::-1], fontsize=7)
ax.set_xlabel('Mean |SHAP|')
ax.set_title('Top 30 Feature Importance (SHAP)', fontsize=14)
ax.invert_yaxis()
plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, 'shap_top30.png')
fig.savefig(fig_path, dpi=150)
print(f"  Saved {fig_path}")

# 2. SHAP summary plot (top 50)
fig, ax = plt.subplots(figsize=(10, 12))
top50 = shap_order[:50]
shap.summary_plot(shap_values[:, top50], X_scaled[:, top50],
                  feature_names=[sel_names[i] for i in top50],
                  max_display=50, show=False, plot_size=(10, 12))
plt.tight_layout()
fig_path2 = os.path.join(RESULTS_DIR, 'shap_summary.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
print(f"  Saved {fig_path2}")
plt.close('all')

# 3. Source contribution pie chart
fig, ax = plt.subplots(figsize=(8, 6))
sources = ['Handcrafted', 'ESM-3', 'UniProt']
source_contrib = [mean_shap[source_indices[s]].sum() / total_shap * 100 for s in sources]
colors = ['#2196F3', '#FF9800', '#4CAF50']
wedges, texts, autotexts = ax.pie(source_contrib, labels=sources, autopct='%1.1f%%',
                                    colors=colors, startangle=90)
for at in autotexts:
    at.set_fontsize(12)
ax.set_title('Feature Importance Share by Source', fontsize=14)
plt.tight_layout()
fig_path3 = os.path.join(RESULTS_DIR, 'shap_source_pie.png')
fig.savefig(fig_path3, dpi=150)
print(f"  Saved {fig_path3}")

# 4. Source composition bar (count and importance per source)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# Count
counts = [len(source_indices[s]) for s in sources]
ax1.bar(sources, counts, color=colors)
ax1.set_ylabel('Number of Features Selected')
ax1.set_title(f'Feature Count by Source (K={K})')
for i, v in enumerate(counts):
    ax1.text(i, v + 10, str(v), ha='center', fontweight='bold')
# Importance
imp_total = [mean_shap[source_indices[s]].sum() for s in sources]
ax2.bar(sources, imp_total, color=colors)
ax2.set_ylabel('Total |SHAP|')
ax2.set_title('Total Importance by Source')
for i, v in enumerate(imp_total):
    ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
fig_path4 = os.path.join(RESULTS_DIR, 'shap_source_bars.png')
fig.savefig(fig_path4, dpi=150)
print(f"  Saved {fig_path4}")

# ---- Save detailed SHAP data ----
shap_data = {
    'feature_names': sel_names,
    'mean_abs_shap': mean_shap.tolist(),
    'shap_order': shap_order.tolist(),
    'source_labels': source_labels,
    'source_contributions': {
        s: {
            'count': len(source_indices[s]),
            'total_shap': float(mean_shap[source_indices[s]].sum()),
            'mean_shap_per_feature': float(mean_shap[source_indices[s]].mean()),
            'share_pct': float(mean_shap[source_indices[s]].sum() / total_shap * 100),
        }
        for s in sources
    },
    'top_features': [
        {'rank': r+1, 'name': sel_names[shap_order[r]],
         'source': source_labels[shap_order[r]],
         'mean_abs_shap': float(mean_shap[shap_order[r]])}
        for r in range(30)
    ],
}

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
json_path = os.path.join(RESULTS_DIR, f'shap_analysis_{timestamp}.json')
with open(json_path, 'w') as f:
    json.dump(shap_data, f, indent=2)
print(f"\nSaved to {json_path}")

# ---- Final Summary for Paper ----
print(f"\n{'#'*70}")
print("# PAPER-READY SHAP SUMMARY")
print(f"{'#'*70}")
print(f"\nSource Contribution (total |SHAP|):")
for s in sources:
    info = shap_data['source_contributions'][s]
    print(f"  {s:<18}: {info['count']:>4} features, "
          f"{info['total_shap']:>8.4f} total SHAP ({info['share_pct']:.1f}%), "
          f"{info['mean_shap_per_feature']:.6f} per feature")

print(f"\nTop 10 most important features:")
for r in range(10):
    f = shap_data['top_features'][r]
    print(f"  {f['rank']:>2}. [{f['source']:<14}] {f['name']:<50} SHAP={f['mean_abs_shap']:.6f}")
