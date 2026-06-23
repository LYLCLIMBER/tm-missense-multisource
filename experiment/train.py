"""
训练与评估：独立测试（10次）和交叉验证（10折）
"""

import math
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

from config import METRICS, GLOBAL_SEED

warnings.filterwarnings("ignore")


def categorical_probas_to_classes(p: np.ndarray) -> np.ndarray:
    """将预测概率矩阵转换为类别预测（取 argmax）。"""
    return np.argmax(p, axis=1)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> Dict[str, float]:
    """
    计算全部评估指标。

    参数:
        y_true: 真实标签（0/1）
        y_pred: 离散预测（0/1）
        y_score: 正类预测概率

    返回:
        包含各指标的字典
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 敏感性 / 召回率
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # 特异性
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # 平衡准确率
    bacc = 0.5 * sn + 0.5 * sp
    # Matthews 相关系数
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0.0
    # AUC
    auc = roc_auc_score(y_true, y_score)
    # 平均精确率 (AP)
    ap = average_precision_score(y_true, y_score)

    return {
        "BACC": bacc,
        "Sn": sn,
        "Sp": sp,
        "MCC": mcc,
        "AUC": auc,
        "AP": ap,
    }


def run_independent_test(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 10,
) -> Dict[str, Tuple[float, float, List[float]]]:
    """
    独立测试（Independence Test）：多次随机 80/20 划分，训练并评估。

    参数:
        model: 未训练的模型实例（内部会 clone，每次迭代重新训练）
        X: 特征矩阵
        y: 标签向量
        n_iter: 重复次数（默认 10）

    返回:
        dict: {指标名: (均值, 标准差, 所有值列表)}
    """
    print(f"\n{'='*50}")
    print(f"独立测试（{n_iter} 次随机 80/20 划分）")
    print(f"{'='*50}")

    all_metrics: Dict[str, List[float]] = {m: [] for m in METRICS}

    for i in range(n_iter):
        # 随机划分（使用不同的随机种子）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i, stratify=y
        )

        # 训练模型
        from sklearn.base import clone

        clf = clone(model)
        clf.fit(X_train, y_train)

        # 预测
        y_score = clf.predict_proba(X_test)[:, 1]
        y_pred = categorical_probas_to_classes(clf.predict_proba(X_test))

        # 计算指标
        metrics = compute_metrics(y_test.values, y_pred, y_score)

        for m in METRICS:
            all_metrics[m].append(metrics[m])

    # 汇总结果
    results = {}
    print(f"\n{'指标':>8} | {'均值':>8} {'标准差':>8}")
    print("-" * 30)
    for m in METRICS:
        mean_val = float(np.mean(all_metrics[m]))
        std_val = float(np.std(all_metrics[m], ddof=1))  # 样本标准差
        results[m] = (mean_val, std_val, all_metrics[m])
        print(f"{m:>8} | {mean_val:.3f}    {std_val:.3f}")
    print("-" * 30)

    return results


def run_cross_validation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 10,
) -> Dict[str, Tuple[float, float, List[float]]]:
    """
    K 折分层交叉验证（Stratified K-Fold Cross Validation）。

    参数:
        model: 未训练的模型实例（内部会 clone，每折重新训练）
        X: 特征矩阵
        y: 标签向量
        n_folds: 折数（默认 10）

    返回:
        dict: {指标名: (均值, 标准差, 所有值列表)}
    """
    print(f"\n{'='*50}")
    print(f"{n_folds} 折分层交叉验证")
    print(f"{'='*50}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=GLOBAL_SEED)
    all_metrics: Dict[str, List[float]] = {m: [] for m in METRICS}

    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 训练模型
        from sklearn.base import clone

        clf = clone(model)
        clf.fit(X_train, y_train)

        # 预测
        y_score = clf.predict_proba(X_test)[:, 1]
        y_pred = categorical_probas_to_classes(clf.predict_proba(X_test))

        # 计算指标
        metrics = compute_metrics(y_test.values, y_pred, y_score)

        print(f"  折 {fold:2d}: BACC={metrics['BACC']:.3f} AUC={metrics['AUC']:.3f} AP={metrics['AP']:.3f}")

        for m in METRICS:
            all_metrics[m].append(metrics[m])

        fold += 1

    # 汇总结果
    results = {}
    print(f"\n{'指标':>8} | {'均值':>8} {'标准差':>8}")
    print("-" * 30)
    for m in METRICS:
        mean_val = float(np.mean(all_metrics[m]))
        std_val = float(np.std(all_metrics[m], ddof=1))
        results[m] = (mean_val, std_val, all_metrics[m])
        print(f"{m:>8} | {mean_val:.3f}    {std_val:.3f}")
    print("-" * 30)

    return results


if __name__ == "__main__":
    # 测试
    from preprocess import run_preprocessing
    from models import get_model

    X, y = run_preprocessing("handcrafted")
    model = get_model("RF")
    results = run_independent_test(model, X, y, n_iter=3)
    for metric, (mean_val, std_val, _) in results.items():
        print(f"{metric}: {mean_val:.3f} +/- {std_val:.3f}")
