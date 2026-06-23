"""
数据预处理：加载数据、删除常量列、标签映射、标准化、数据划分
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple

from config import DATA_FILE, GLOBAL_SEED, FEATURE_SETS, FEATURE_SET_NAMES, PLM_FEATURE_SETS, ESM_FEATURE_FILE, PROT_FEATURE_FILE, ESM3_79_FEATURE_FILE


def load_data() -> pd.DataFrame:
    """
    加载 data.xlsx 文件。
    返回原始 DataFrame。
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"数据文件未找到: {DATA_FILE}")
    df = pd.read_excel(DATA_FILE)
    return df


def remove_separator_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    删除 separator 常量列（列名为 'separator'，所有行值为 150.788）。
    该列是网格搜索时的占位列，不包含任何预测信息。
    """
    if "separator" in df.columns:
        df = df.drop(columns=["separator"])
        print("[数据预处理] 已删除 separator 常量列")
    else:
        print("[数据预处理] 未找到 separator 列，跳过删除")
    return df


def map_labels(y: pd.Series) -> pd.Series:
    """
    标签映射：1（致病）→ 1, 2（良性）→ 0

    原始 BorodaTM 数据集中 class=1 有 392 个样本，对应论文描述中的
    disease-associated mutations；class=2 有 154 个样本，对应 neutral mutations。
    """
    y = y.replace({1: 1, 2: 0})
    return y


def get_features_and_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    从 DataFrame 提取特征矩阵 X 和标签向量 y。
    X: 从第3列（索引2）开始的所有列（不含 'name' 和 'class'）
    y: 第2列（索引1），即 'class' 列
    """
    X = df.iloc[:, 2:]
    y = df.iloc[:, 1]
    y = map_labels(y)
    return X, y


def print_data_info(X: pd.DataFrame, y: pd.Series) -> None:
    """
    打印数据基本信息：样本量、特征维度、类别分布。
    """
    print("=" * 50)
    print("数据集基本信息")
    print("=" * 50)
    print(f"  样本数: {X.shape[0]}")
    print(f"  特征维度: {X.shape[1]}")
    print(f"  类别分布:")
    value_counts = y.value_counts().sort_index()
    for label, count in value_counts.items():
        label_name = "良性(0)" if label == 0 else "致病(1)"
        print(f"    {label_name}: {count} ({count / len(y) * 100:.1f}%)")
    print("=" * 50)


def load_plm_features(feature_set_name: str, df_index) -> pd.DataFrame:
    """
    加载 PLM 特征（从 .npy 文件），返回与 df_index 索引一致的 DataFrame。

    参数:
        feature_set_name: 特征集名称（如 'esm_only', 'prot_only', 'plm_fusion', 'esm3_only', 'esm3_fusion'）
        df_index: 原始数据 DataFrame 的索引，用于结果对齐

    返回:
        DataFrame: PLM 特征矩阵
    """
    if feature_set_name in ("esm3_only", "esm3_fusion"):
        if not ESM3_79_FEATURE_FILE.exists():
            raise FileNotFoundError(f"ESM-3特征文件未找到: {ESM3_79_FEATURE_FILE}")
        esm3 = np.load(ESM3_79_FEATURE_FILE)  # (546, 2560)
        X = pd.DataFrame(esm3, index=df_index)
        print(f"[PLM加载] ESM-3 Layer79特征: {X.shape}")
        return X

    if not ESM_FEATURE_FILE.exists():
        raise FileNotFoundError(f"ESM特征文件未找到: {ESM_FEATURE_FILE}")
    if not PROT_FEATURE_FILE.exists():
        raise FileNotFoundError(f"ProtT5特征文件未找到: {PROT_FEATURE_FILE}")

    esm = np.load(ESM_FEATURE_FILE)    # (546, 1280)
    prot = np.load(PROT_FEATURE_FILE)  # (546, 1024)

    if feature_set_name == "esm_only":
        X = pd.DataFrame(esm, index=df_index)
        print(f"[PLM加载] ESM-1v特征: {X.shape}")
    elif feature_set_name == "prot_only":
        X = pd.DataFrame(prot, index=df_index)
        print(f"[PLM加载] ProtT5特征: {X.shape}")
    elif feature_set_name == "plm_fusion":
        X = pd.DataFrame(np.hstack([esm, prot]), index=df_index)
        print(f"[PLM加载] ESM+ProtT5融合特征: {X.shape}")
    elif feature_set_name == "all_fusion_plm":
        raise ValueError("all_fusion_plm should use run_preprocessing with all_fusion mode")
    else:
        raise ValueError(
            f"未知的 PLM 特征集: '{feature_set_name}'。"
            f"可选: esm_only, prot_only, plm_fusion, esm3_only, esm3_fusion"
        )

    return X


def select_features_by_set(
    X: pd.DataFrame, feature_set_name: str
) -> pd.DataFrame:
    """
    根据特征集名称从 X 中选择对应的列。
    特征集定义见 config.FEATURE_SETS。
    """
    if feature_set_name not in FEATURE_SETS:
        raise ValueError(
            f"未知的特征集名称: '{feature_set_name}'。"
            f"可选值: {list(FEATURE_SETS.keys())}"
        )

    selection = FEATURE_SETS[feature_set_name]
    set_name_cn = FEATURE_SET_NAMES.get(feature_set_name, feature_set_name)

    if selection == "all":
        print(f"[特征选择] 使用全部特征 ({X.shape[1]} 维)")
        return X
    elif isinstance(selection, list):
        # 检查索引是否在有效范围内
        max_idx = X.shape[1] - 1
        invalid_idx = [i for i in selection if i > max_idx]
        if invalid_idx:
            raise IndexError(
                f"特征集 '{feature_set_name}' 包含无效索引 {invalid_idx}，"
                f"X 的列范围为 0..{max_idx}"
            )
        X_selected = X.iloc[:, selection]
        print(f"[特征选择] 使用 {set_name_cn} ({X_selected.shape[1]} 维)")
        return X_selected
    else:
        raise TypeError(f"FEATURE_SETS['{feature_set_name}'] 类型不支持: {type(selection)}")


def standardize_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    使用 StandardScaler 对特征进行标准化（Z-score）。
    在训练集上拟合 scaler，然后转换训练集和测试集。
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"[标准化] 训练集: {X_train_scaled.shape}, 测试集: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, scaler


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = GLOBAL_SEED,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    将数据划分为训练集和测试集，并可选进行标准化。

    参数:
        X: 特征矩阵
        y: 标签向量
        test_size: 测试集比例（默认 0.2）
        random_state: 随机种子
        scale: 是否进行标准化（默认 True）

    返回:
        X_train, X_test, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scale:
        X_train, X_test, scaler = standardize_features(X_train, X_test)
    else:
        scaler = None

    return X_train, X_test, y_train, y_test, scaler


def run_preprocessing(feature_set: str = "all_fusion") -> Tuple[pd.DataFrame, pd.Series]:
    """
    运行完整预处理流程：
    1. 加载数据
    2. 删除 separator 常量列
    3. 提取标签（标签映射）
    4. 按特征集名称选择/加载特征
       - 对于 PLM 特征集（esm_only, prot_only, plm_fusion）：从 .npy 文件动态加载
       - 对于 all_fusion（特殊处理）：手工+ESM+ProtT5 三路拼接
       - 其他：从 data.xlsx 按列索引选择
    5. 打印数据信息

    参数:
        feature_set: 特征集名称

    返回:
        X, y
    """
    print("\n" + "=" * 50)
    print("数据预处理")
    print("=" * 50)

    # 1. 加载数据
    df = load_data()
    print(f"[加载数据] 原始形状: {df.shape}")

    # 2. 删除 separator
    df = remove_separator_column(df)
    print(f"[删除常量列] 当前形状: {df.shape}")

    # 3. 提取标签
    y = df.iloc[:, 1]
    y = map_labels(y)
    print(f"[提取标签] y: {y.shape}")

    # 4. 按特征集名称处理
    if feature_set in PLM_FEATURE_SETS:
        # PLM 特征动态加载
        X = load_plm_features(feature_set, df.index)
        # esm3_fusion: 额外拼接手工特征
        if feature_set == "esm3_fusion":
            X_hand = df.iloc[:, 2:]
            X = pd.DataFrame(
                np.hstack([X_hand.values, X.values]),
                index=df.index,
            )
            print(f"[特征选择] 手工+ESM-3 Layer79融合 ({X.shape[1]}维)")
    elif feature_set == "all_fusion":
        # 特殊处理: 手工+ESM+ProtT5 全融合
        X_hand = df.iloc[:, 2:]  # 250维手工特征
        esm = np.load(ESM_FEATURE_FILE)
        prot = np.load(PROT_FEATURE_FILE)
        X = pd.DataFrame(
            np.hstack([X_hand.values, esm, prot]),
            index=df.index,
        )
        print(f"[特征选择] 使用手工+ESM+ProtT5全融合 ({X.shape[1]}维)")
    else:
        # 原流程: 从 data.xlsx 按列索引选择
        X, _ = get_features_and_labels(df)
        X = select_features_by_set(X, feature_set)

    # 5. 打印信息
    print_data_info(X, y)

    return X, y


if __name__ == "__main__":
    # 测试：使用全部特征
    X, y = run_preprocessing("all_fusion")
    print(f"\nX 类型: {type(X)}, y 类型: {type(y)}")
    print(f"X 前5行索引: {X.index[:5].tolist()}")
    print(f"y 前10个值: {y.head(10).tolist()}")
