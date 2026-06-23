"""
模型定义：XGBoost、RandomForest、SVM、MLP
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from config import MODEL_PARAMS


def get_model(model_name: str, **kwargs):
    """
    根据模型名称返回配置好的 sklearn 兼容分类器。

    参数:
        model_name: 模型名称，可选 'XGBoost', 'RF', 'SVM', 'MLP'
        **kwargs: 覆盖默认超参数的额外参数

    返回:
        配置好的模型实例
    """
    model_name = model_name.strip()

    if model_name not in MODEL_PARAMS:
        raise ValueError(
            f"未知模型: '{model_name}'。可选: {list(MODEL_PARAMS.keys())}"
        )

    # 从配置中复制默认参数
    params = dict(MODEL_PARAMS[model_name])

    # 用用户提供的参数覆盖
    params.update(kwargs)

    if model_name == "XGBoost":
        from xgboost import XGBClassifier

        model = XGBClassifier(**params)

    elif model_name == "RF":
        model = RandomForestClassifier(**params)

    elif model_name == "SVM":
        model = SVC(**params)

    elif model_name == "MLP":
        model = MLPClassifier(**params)

    return model


def list_available_models() -> list:
    """返回可用模型名称列表。"""
    return list(MODEL_PARAMS.keys())


if __name__ == "__main__":
    print("可用模型:", list_available_models())
    for name in list_available_models():
        model = get_model(name)
        print(f"  {name}: {model}")
        print(f"    参数: {model.get_params()}")
