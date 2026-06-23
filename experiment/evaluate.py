"""
评估工具：DeLong 检验、结果格式化、ROC/PR 曲线绘制、CSV 保存
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免无 GUI 时报错
import matplotlib.pyplot as plt

from config import RESULTS_DIR, METRICS


# ========== DeLong 检验 ==========


def _delong_roc_variance(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    DeLong 法计算 AUC 的方差。

    参数:
        y_true: 真实标签（0/1）
        y_score: 预测概率

    返回:
        auc, variance
    """
    from scipy.stats import norm

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    n = n_pos + n_neg

    # 按分数排序
    idx = np.argsort(y_score)
    y_true_sorted = y_true[idx]
    y_score_sorted = y_score[idx]

    # 计算秩
    rank = np.argsort(idx)

    # 正例和负例的秩
    pos_rank = rank[y_true == 1]
    neg_rank = rank[y_true == 0]

    # AUC
    auc = (np.mean(pos_rank) - (n_pos + 1) / 2) / n_neg

    # 计算方差
    pos_score = y_score[y_true == 1]
    neg_score = y_score[y_true == 0]

    # theta1: 正例对之间
    pos_theta = np.zeros((n_pos, n_pos))
    for i in range(n_pos):
        for j in range(n_pos):
            if i != j:
                pos_theta[i, j] = 1.0 if pos_score[i] > pos_score[j] else 0.0
                if pos_score[i] == pos_score[j]:
                    pos_theta[i, j] = 0.5
    V10 = np.var(pos_theta.sum(axis=1), ddof=1) / (n_pos - 1) if n_pos > 1 else 0.0

    # theta0: 负例对之间
    neg_theta = np.zeros((n_neg, n_neg))
    for i in range(n_neg):
        for j in range(n_neg):
            if i != j:
                neg_theta[i, j] = 1.0 if neg_score[i] > neg_score[j] else 0.0
                if neg_score[i] == neg_score[j]:
                    neg_theta[i, j] = 0.5
    V01 = np.var(neg_theta.sum(axis=1), ddof=1) / (n_neg - 1) if n_neg > 1 else 0.0

    variance = V10 / n_pos + V01 / n_neg
    return auc, variance


def delong_test(
    y_true: np.ndarray,
    y_score_1: np.ndarray,
    y_score_2: np.ndarray,
) -> Tuple[float, float]:
    """
    DeLong 检验比较两个模型的 AUC 是否有显著差异。

    参数:
        y_true: 真实标签
        y_score_1: 模型1的预测概率
        y_score_2: 模型2的预测概率

    返回:
        z_statistic, p_value（双侧检验）
    """
    from scipy.stats import norm

    auc1, var1 = _delong_roc_variance(y_true, y_score_1)
    auc2, var2 = _delong_roc_variance(y_true, y_score_2)

    # 计算协方差
    n = len(y_true)
    n_pos = np.sum(y_true == 1)
    n_neg = n - n_pos

    # 计算两个评分在正例和负例上的秩
    rank_1 = np.argsort(np.argsort(y_score_1))
    rank_2 = np.argsort(np.argsort(y_score_2))

    pos_rank_1 = rank_1[y_true == 1]
    neg_rank_1 = rank_1[y_true == 0]
    pos_rank_2 = rank_2[y_true == 1]
    neg_rank_2 = rank_2[y_true == 0]

    cov = (np.cov(pos_rank_1, pos_rank_2, ddof=1)[0, 1] / n_pos +
           np.cov(neg_rank_1, neg_rank_2, ddof=1)[0, 1] / n_neg)

    var_diff = var1 + var2 - 2 * cov
    if var_diff <= 0:
        var_diff = 1e-10

    z = (auc1 - auc2) / np.sqrt(var_diff)
    p = 2 * (1 - norm.cdf(abs(z)))

    return z, p


# ========== 结果格式化 ==========


def format_metrics_table(
    results: Dict[str, Dict[str, Tuple[float, float, List[float]]]],
    title: str = "",
) -> str:
    """
    将多个模型的评估结果格式化为表格字符串。

    参数:
        results: {模型名: {指标: (均值, 标准差, 列表)}}
        title: 表格标题

    返回:
        格式化的表格字符串
    """
    lines = []
    if title:
        lines.append(f"\n{'='*60}")
        lines.append(f"  {title}")
        lines.append(f"{'='*60}")

    # 表头
    header = f"{'模型':<12}"
    for m in METRICS:
        header += f" | {m:>8}"
    lines.append(header)
    lines.append("-" * len(header))

    # 数据行
    for model_name, metrics in results.items():
        row = f"{model_name:<12}"
        for m in METRICS:
            if m in metrics:
                mean_val, std_val, _ = metrics[m]
                row += f" | {mean_val:.3f}±{std_val:.3f}"
            else:
                row += f" | {'':>8}"
        lines.append(row)

    lines.append("=" * 60)
    return "\n".join(lines)


def print_comparison_table(
    ind_results: Dict[str, Dict],
    cv_results: Dict[str, Dict],
) -> None:
    """
    打印独立测试和交叉验证的对比表格。

    参数:
        ind_results: 独立测试结果
        cv_results: 交叉验证结果
    """
    print("\n" + "=" * 70)
    print("  综合结果对比")
    print("=" * 70)

    header = f"{'模型':<12} | {'方式':<6}"
    for m in METRICS:
        header += f" | {m:>8}"
    print(header)
    print("-" * len(header))

    all_models = set(list(ind_results.keys()) + list(cv_results.keys()))
    for model_name in sorted(all_models):
        if model_name in ind_results:
            row = f"{model_name:<12} | {'独立':<6}"
            for m in METRICS:
                if m in ind_results[model_name]:
                    mean_val, std_val, _ = ind_results[model_name][m]
                    row += f" | {mean_val:.3f}±{std_val:.3f}"
                else:
                    row += f" | {'':>8}"
            print(row)

        if model_name in cv_results:
            row = f"{model_name:<12} | {'交叉':<6}"
            for m in METRICS:
                if m in cv_results[model_name]:
                    mean_val, std_val, _ = cv_results[model_name][m]
                    row += f" | {mean_val:.3f}±{std_val:.3f}"
                else:
                    row += f" | {'':>8}"
            print(row)

        print("-" * len(header))
    print()


# ========== ROC / PR 曲线绘制 ==========


def plot_roc_curves(
    roc_data: Dict[str, Dict],
    save_path: Optional[Path] = None,
    title: str = "ROC Curve Comparison",
) -> None:
    """
    绘制多个模型的 ROC 曲线对比图。

    参数:
        roc_data: {模型名: {'fpr': np.array, 'tpr': np.array, 'auc': float}}
        save_path: 保存路径（可选）
        title: 图表标题
    """
    plt.figure(figsize=(8, 6))

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
    for idx, (model_name, data) in enumerate(roc_data.items()):
        color = colors[idx % len(colors)]
        plt.plot(
            data["fpr"],
            data["tpr"],
            color=color,
            lw=2,
            label=f"{model_name} (AUC={data['auc']:.3f})",
        )

    # 对角线
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", alpha=0.7)

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)

    if save_path:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[图表保存] {save_path}")
    plt.close()


def plot_pr_curves(
    pr_data: Dict[str, Dict],
    save_path: Optional[Path] = None,
    title: str = "PR Curve Comparison",
) -> None:
    """
    绘制多个模型的 PR 曲线对比图。

    参数:
        pr_data: {模型名: {'recall': np.array, 'precision': np.array, 'ap': float}}
        save_path: 保存路径（可选）
        title: 图表标题
    """
    plt.figure(figsize=(8, 6))

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
    for idx, (model_name, data) in enumerate(pr_data.items()):
        color = colors[idx % len(colors)]
        plt.plot(
            data["recall"],
            data["precision"],
            color=color,
            lw=2,
            label=f"{model_name} (AP={data['ap']:.3f})",
        )

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)

    if save_path:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[图表保存] {save_path}")
    plt.close()


# ========== 结果保存 ==========


def save_results_to_csv(
    results: Dict[str, Dict[str, Tuple[float, float, List[float]]]],
    save_path: Path,
    experiment_name: str = "",
) -> None:
    """
    将评估结果保存为 CSV 文件。

    CSV 格式：
        模型, 指标, 均值, 标准差

    参数:
        results: {模型名: {指标: (均值, 标准差, 列表)}}
        save_path: CSV 保存路径
        experiment_name: 实验名称（用于注释）
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if experiment_name:
            writer.writerow([f"# {experiment_name}"])
        writer.writerow(["模型", "指标", "均值", "标准差"])

        for model_name in sorted(results.keys()):
            for m in METRICS:
                if m in results[model_name]:
                    mean_val, std_val, _ = results[model_name][m]
                    writer.writerow([model_name, m, f"{mean_val:.4f}", f"{std_val:.4f}"])

    print(f"[结果保存] {save_path}")


if __name__ == "__main__":
    print("evaluate.py - 评估工具模块")
    print("可用函数: delong_test, format_metrics_table, print_comparison_table, "
          "plot_roc_curves, plot_pr_curves, save_results_to_csv")
