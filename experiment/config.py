"""
配置文件：路径、随机种子、模型超参数、特征设定
"""

import sys
from pathlib import Path

# 项目根目录（experiment/ 的父目录）
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据目录（论文4的附件位置）
# 原路径有中文编码问题，现数据文件位于项目根目录
DATA_DIR = BASE_DIR

# 特征与结果目录
FEATURES_DIR = BASE_DIR / "experiment" / "features"
RESULTS_DIR = BASE_DIR / "experiment" / "results"

# 数据文件路径
DATA_FILE = DATA_DIR / "data.xlsx"

# PLM 特征文件路径（使用全长序列提取的特征）
ESM_FEATURE_FILE = FEATURES_DIR / "X_esm_full.npy"
PROT_FEATURE_FILE = FEATURES_DIR / "X_prot_full.npy"
ESM3_79_FEATURE_FILE = FEATURES_DIR / "esm3_79" / "X_esm3_79.npy"

# 全局随机种子
GLOBAL_SEED = 42

# ========== 模型超参数（基于论文4网格搜索结果） ==========

MODEL_PARAMS = {
    "XGBoost": {
        "n_estimators": 80,
        "max_depth": 2,
        "learning_rate": 0.2,
        "objective": "binary:logistic",
        "random_state": GLOBAL_SEED,
        "verbosity": 0,
    },
    "RF": {
        "n_estimators": 160,
        "max_depth": 4,
        "random_state": GLOBAL_SEED,
    },
    "SVM": {
        "C": 0.1,
        "kernel": "linear",
        "probability": True,
        "random_state": GLOBAL_SEED,
    },
    "MLP": {
        "hidden_layer_sizes": (320,),
        "early_stopping": True,
        "max_iter": 3000,
        "n_iter_no_change": 40,
        "random_state": GLOBAL_SEED,
    },
}

# ========== 特征设定 ==========
# 各设定对应的列索引范围（基于 df.iloc[:, 2:] 的特征矩阵）
# 数据列结构（从原始 df 第2列开始）：
#   索引 0..104:  handcrafted 特征（redundancy ~ mut_ncontacts_special）
#   索引 105..244: PSSM 特征（pssm1 ~ pssm140）
#   索引 245..250: 保守性打分特征（fathmm_Score ~ SIFT_SCORE）
#
# 注：separator 列位于 handcrafted 内部（原 df 第71列，BS），
#    在 preprocess.py 中会被单独删除。

FEATURE_SETS = {
    "handcrafted": list(range(0, 105)),              # 手工特征（不含 separator）
    "pssm_only": list(range(105, 245)),              # PSSM 特征
    "conservation_only": list(range(245, 251)),      # 保守性打分
    "handcrafted_pssm": list(range(0, 105)) + list(range(105, 245)),  # 手工 + PSSM
    "all_fusion": "all",                              # 全部特征
}

# PLM 特征集名称（动态加载的特征集，不使用列索引）
# 这些名称在 preprocess.py 中被识别并触发 PLM 特征加载
PLM_FEATURE_SETS = {"esm_only", "prot_only", "plm_fusion", "esm3_only", "esm3_fusion"}

# 特征设定中文名（用于输出）
FEATURE_SET_NAMES = {
    "handcrafted": "手工特征(105维)",
    "pssm_only": "PSSM特征",
    "conservation_only": "保守性打分",
    "handcrafted_pssm": "手工+PSSM特征融合",
    "all_fusion": "全部特征融合",
    "esm_only": "ESM-1v特征",
    "prot_only": "ProtT5特征",
    "plm_fusion": "ESM+ProtT5融合",
    "esm3_only": "ESM-3 Layer79(2560维)",
    "esm3_fusion": "手工+ESM-3 Layer79融合",
}

# 评估指标
METRICS = ["BACC", "Sn", "Sp", "MCC", "AUC", "AP"]

if __name__ == "__main__":
    print(f"BASE_DIR = {BASE_DIR}")
    print(f"DATA_FILE = {DATA_FILE}")
    print(f"DATA_FILE exists = {DATA_FILE.exists()}")
