"""
utils_reward.py
===============
奖励函数工具函数。包含可插拔的长度惩罚函数等。

所有长度惩罚函数签名统一为:
    func(num1: int, num2: int, **kwargs) -> float
    返回范围 [-1, 0]，0 表示无惩罚，-1 表示最大惩罚。

注册新函数: 在 LENGTH_PENALTY_FUNCTIONS 字典中添加即可。
"""

import math
from typing import Callable, Dict


# ================================================================
# 长度惩罚函数
# ================================================================

def default_length_penalty(
    num1: int,
    num2: int,
    tolerance_ratio: float = 0.5,
    max_penalty_ratio: float = 3.0,
) -> float:
    """
    默认长度惩罚函数（分段线性）。

    规则:
      - num3 = |num1 - num2|
      - num3 <= num1 * tolerance_ratio  →  惩罚 = 0
      - num3 >= num1 * max_penalty_ratio →  惩罚 = -1
      - 中间区域线性插值

    Args:
        num1: 标准 completion 中 nodes 数量
        num2: 预测 completion_hat 中 nodes 数量
        tolerance_ratio: 容忍比例（默认 0.5）
        max_penalty_ratio: 达到最大惩罚的比例（默认 3.0）

    Returns:
        长度惩罚值，范围 [-1, 0]
    """
    if num1 <= 0:
        # 标准答案无节点，预测有节点 → 最大惩罚；预测也无节点 → 无惩罚
        return -1.0 if num2 > 0 else 0.0

    num3 = abs(num1 - num2)
    tol = num1 * tolerance_ratio
    max_tol = num1 * max_penalty_ratio

    if num3 <= tol:
        return 0.0
    elif num3 >= max_tol:
        return -1.0
    else:
        # 线性插值: tol → 0, max_tol → -1
        ratio = (num3 - tol) / (max_tol - tol)
        return -ratio


def sigmoid_length_penalty(
    num1: int,
    num2: int,
    tolerance_ratio: float = 0.5,
    steepness: float = 2.0,
) -> float:
    """
    Sigmoid 平滑版长度惩罚函数。

    与 default_length_penalty 类似，但使用 sigmoid 平滑过渡，
    避免分段线性的"拐点"对训练造成突变。

    Args:
        num1: 标准 completion 中 nodes 数量
        num2: 预测 completion_hat 中 nodes 数量
        tolerance_ratio: 容忍比例
        steepness: sigmoid 陡峭度（越大越接近硬截断）

    Returns:
        长度惩罚值，范围 [-1, 0]
    """
    if num1 <= 0:
        return -1.0 if num2 > 0 else 0.0

    num3 = abs(num1 - num2)
    tol = num1 * tolerance_ratio

    if num3 <= tol:
        return 0.0

    # 将 (num3 - tol) / num1 通过 sigmoid 映射到 [-1, 0]
    x = (num3 - tol) / max(num1, 1) * steepness
    sigmoid_val = 1.0 / (1.0 + math.exp(-x))
    # sigmoid 对 x>=0 输出 [0.5, 1.0]，映射到 [-1, 0]
    return -(sigmoid_val - 0.5) * 2.0


# ================================================================
# 函数注册表
# ================================================================

LENGTH_PENALTY_FUNCTIONS: Dict[str, Callable] = {
    "default_length_penalty": default_length_penalty,
    "sigmoid_length_penalty": sigmoid_length_penalty,
}
