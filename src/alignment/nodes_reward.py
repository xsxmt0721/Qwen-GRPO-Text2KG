"""
nodes_reward.py
===============
节点提取任务奖励函数 (rule-based)，用于 GRPO 训练中评估模型输出质量。

奖励组成 (四项加权求和):
  1. 结构得分      — JSON 能否解析，关键字 nodes/name/type 是否正确
  2. 长度惩罚      — 预测节点数与标准节点数的偏差
  3. 文本匹配度    — 预测节点能否在输入文本中找到 (L0→L1→L1.5→L2)
  4. 数据集匹配度  — 预测节点与标准节点集 node.json 的相似度

用法:
    from src.alignment.nodes_reward import NodeRewardCalculator, compute_node_reward

    # 方式1: 单例
    score = compute_node_reward(completion_hat, completion, text)

    # 方式2: 实例
    calc = NodeRewardCalculator("config/reward.yaml")
    result = calc.calculate_reward(completion_hat, completion, text)
"""

import json
import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import yaml

# ── 项目根路径 ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils_text import (
    _match_level0,
    _match_level1,
    _match_level1_5,
    _match_level2,
    _tokenize,
    _normalize,
)
from src.utils.utils_reward import LENGTH_PENALTY_FUNCTIONS

logger = logging.getLogger(__name__)


# ================================================================
# 辅助: JSON 提取与修复
# ================================================================

def _extract_json_from_completion(completion_str: str) -> Optional[str]:
    """
    从 completion 字符串中提取 JSON 内容。

    处理:
      - ```json ... ```  包裹格式
      - 裸 {...} 或 [...] JSON
    """
    if not completion_str:
        return None

    s = completion_str.strip()

    # ① 提取 ```json ... ``` 中的内容
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', s, re.DOTALL)
    if m:
        return m.group(1).strip()

    # ② 找到第一个 { 或 [，截取到末尾
    for start_char in ('{', '['):
        idx = s.find(start_char)
        if idx != -1:
            return s[idx:].strip()

    return s


def _try_fix_truncated_json(json_str: str) -> Optional[str]:
    """
    尝试修复截断的 JSON。

    策略 (仅闭合，不新增内容):
      1. 如果截断在字符串中间，闭合引号
      2. 按嵌套顺序补齐缺失的 ] 和 }（后进先出）
    """
    if not json_str or not json_str.strip():
        return None

    s = json_str.strip()

    # 去掉 ```json 前缀和 ``` 后缀（如果存在）
    s = re.sub(r'^```(?:json)?\s*\n?', '', s)
    s = re.sub(r'\n?\s*```\s*$', '', s)

    # ── 判断是否在字符串内截断，并同时记录括号栈（忽略字符串内的括号）──
    in_string = False
    bracket_stack: list = []  # 记录期待闭合的括号，栈顶 = 最近打开的

    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '\\':
            i += 2
            continue
        if ch == '"':
            in_string = not in_string
            i += 1
            continue
        if not in_string:
            if ch == '{':
                bracket_stack.append('}')
            elif ch == '[':
                bracket_stack.append(']')
            elif ch == '}' or ch == ']':
                if bracket_stack and bracket_stack[-1] == ch:
                    bracket_stack.pop()
        i += 1

    # 如果在字符串中间截断，补闭合引号
    if in_string:
        s += '"'

    # 按栈的逆序补齐缺失的括号（后进先出）
    s += ''.join(reversed(bracket_stack))

    return s


# ================================================================
# 辅助: JSON 解析与校验
# ================================================================

def _try_parse_json(json_str: str) -> Tuple[Optional[List[Dict]], bool, Optional[str]]:
    """
    尝试将 JSON 字符串解析为节点列表。

    Args:
        json_str: 待解析的 JSON 字符串

    Returns:
        (nodes_list, is_correct_keys, error_reason)
        - nodes_list: 解析出的节点列表，失败为 None
        - is_correct_keys: 关键字 nodes/name/type 是否全部正确
        - error_reason: 失败原因（成功时为 None）
    """
    # ── Step 1: json.loads ──
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, False, f"JSONDecodeError: {e}"

    # ── Step 2: 定位 nodes 列表 ──
    nodes_raw = None
    is_correct_keys = True

    if isinstance(data, dict):
        if "nodes" in data:
            nodes_raw = data["nodes"]
        else:
            # 容错: 尝试大小写变体 / 常见别名
            alt_keys = ["node", "Node", "Nodes", "NODES", "entities", "entity", "Entity"]
            for k in alt_keys:
                if k in data:
                    nodes_raw = data[k]
                    is_correct_keys = False
                    break
            if nodes_raw is None:
                return None, False, "missing 'nodes' key in JSON object"
    elif isinstance(data, list):
        nodes_raw = data
        is_correct_keys = False  # 直接给列表也算关键字不标准
    else:
        return None, False, "JSON root is not dict or list"

    if not isinstance(nodes_raw, list):
        return None, False, "'nodes' is not a list"

    # ── Step 3: 提取每个节点的 name / type ──
    nodes = []
    for item in nodes_raw:
        if not isinstance(item, dict):
            is_correct_keys = False
            continue

        # name
        name = item.get("name") or item.get("Name") or item.get("NAME")
        if name is None or not isinstance(name, str) or not name.strip():
            is_correct_keys = False
            continue
        if "name" not in item:
            is_correct_keys = False

        # type
        node_type = item.get("type") or item.get("Type") or item.get("TYPE")
        if node_type is None:
            is_correct_keys = False
            node_type = ""
        if not isinstance(node_type, str):
            node_type = str(node_type)
        if "type" not in item:
            is_correct_keys = False

        nodes.append({
            "name": name.strip(),
            "type": node_type.strip(),
        })

    # 空节点列表是合法结构 (对应 {"nodes": []})，同样返回
    return nodes, is_correct_keys, None


# ================================================================
# 节点奖励计算器
# ================================================================

class NodeRewardCalculator:
    """
    节点提取奖励计算器。

    初始化时加载:
      - config/reward.yaml  中的各项参数
      - Data/data/node.json 标准节点集
    """

    def __init__(self, config_path: str = "config/reward.yaml"):
        # ── 加载配置 ──
        config_full_path = PROJECT_ROOT / config_path
        if not config_full_path.exists():
            raise FileNotFoundError(f"Config not found: {config_full_path}")
        with open(config_full_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # ── 结构得分系数 ──
        self.struct_cfg = self.config["structure"]

        # ── 长度惩罚函数 ──
        self.lp_cfg = self.config["length_penalty"]
        lp_fn_name = self.lp_cfg["function"]
        if lp_fn_name not in LENGTH_PENALTY_FUNCTIONS:
            available = list(LENGTH_PENALTY_FUNCTIONS.keys())
            raise ValueError(
                f"Unknown length_penalty function '{lp_fn_name}'. "
                f"Available: {available}"
            )
        self._length_penalty_fn = LENGTH_PENALTY_FUNCTIONS[lp_fn_name]

        # ── 文本匹配参数 ──
        self.tm_cfg = self.config["text_matching"]

        # ── 数据集匹配参数 ──
        self.dm_cfg = self.config["dataset_matching"]
        self._std_nodes_dict: Dict[str, Dict] = {}   # name → merged node
        self._std_nodes_list: List[Dict] = []          # 用于 L2 遍历
        self._std_tokens_cache: Dict[str, set] = {}    # name → 预 tokenize 集合 (L2 加速)
        self._std_token_list: List[Tuple[Dict, set]] = []  # (node_dict, token_set) 并行列表
        self._load_standard_nodes()

        # ── 权重 ──
        self.weights = self.config["weights"]

        logger.info("NodeRewardCalculator initialized (struct=%.2f/%.2f, "
                     "lp=%s, tm_f1, dm=%s)",
                     self.weights["structure"], self.weights["length_penalty"],
                     lp_fn_name, self.dm_cfg["mode"])

    # ================================================================
    # 标准节点集加载
    # ================================================================

    def _load_standard_nodes(self) -> None:
        """加载 node.json 并构建 name→node 哈希表（合并重名节点）。"""
        node_json_path = PROJECT_ROOT / self.dm_cfg["node_json_path"]

        if not node_json_path.exists():
            logger.warning("Standard node file not found: %s. "
                           "Dataset matching will return 0.", node_json_path)
            return

        with open(node_json_path, 'r', encoding='utf-8') as f:
            raw_nodes = json.load(f)

        for i, node in enumerate(raw_nodes):
            name = node.get("name", "")
            if not name:
                continue

            if name in self._std_nodes_dict:
                # 合并重名节点：type / use / disease 列表去重
                existing = self._std_nodes_dict[name]
                for field in ("type", "use", "disease"):
                    existing[field] = list(set(
                        existing.get(field, []) + node.get(field, [])
                    ))
            else:
                self._std_nodes_dict[name] = {
                    "name": name,
                    "type": list(node.get("type", [])),
                    "use": list(node.get("use", [])),
                    "disease": list(node.get("disease", [])),
                    "_id": i,
                }

        self._std_nodes_list = list(self._std_nodes_dict.values())

        # ── 预 tokenize 所有标准节点名 (避免 L2 匹配时重复计算) ──
        for std_node in self._std_nodes_list:
            name = std_node["name"]
            tokens = _tokenize(name)
            self._std_tokens_cache[name] = tokens
            self._std_token_list.append((std_node, tokens))

        logger.info("Loaded %d unique standard nodes from %d raw entries (%s) [pre-tokenized]",
                     len(self._std_nodes_dict), len(raw_nodes), node_json_path.name)

    # ================================================================
    # 结构解析
    # ================================================================

    def _parse_completion(
        self, completion_str: str
    ) -> Tuple[Optional[List[Dict]], float]:
        """
        解析 completion 字符串，返回 (节点列表, 结构得分)。

        解析顺序:
          1. 直接 json.loads
          2. 截断修复后 json.loads
          3. 均失败 → (None, 0.0)
        """
        # ── 提取 JSON 内容 ──
        json_content = _extract_json_from_completion(completion_str)
        if json_content is None:
            logger.debug("No JSON content found in completion")
            return None, self.struct_cfg["invalid"]

        # ── 直接解析 ──
        nodes, is_correct_keys, _ = _try_parse_json(json_content)
        if nodes is not None:
            score = (
                self.struct_cfg["valid_json_correct_keys"] if is_correct_keys
                else self.struct_cfg["valid_json_wrong_keys"]
            )
            return nodes, score

        # ── 截断修复后解析 ──
        fixed = _try_fix_truncated_json(json_content)
        if fixed is not None and fixed != json_content:
            nodes, is_correct_keys, _ = _try_parse_json(fixed)
            if nodes is not None:
                score = (
                    self.struct_cfg["truncated_fixed_correct_keys"] if is_correct_keys
                    else self.struct_cfg["truncated_fixed_wrong_keys"]
                )
                return nodes, score

        # ── 彻底失败 ──
        return None, self.struct_cfg["invalid"]

    # ================================================================
    # 长度惩罚
    # ================================================================

    def _compute_length_penalty(self, num1: int, num2: int) -> float:
        """调用配置的长度惩罚函数。"""
        # 从 lp_cfg 中提取函数所需的 kwargs
        fn_kwargs = {}
        for key in ("tolerance_ratio", "max_penalty_ratio", "steepness"):
            if key in self.lp_cfg:
                fn_kwargs[key] = self.lp_cfg[key]

        return self._length_penalty_fn(num1, num2, **fn_kwargs)

    # ================================================================
    # 文本匹配度 (预测节点 vs 标准节点)
    # ================================================================

    def _compute_text_matching(
        self, pred_nodes: List[Dict], std_nodes: List[Dict]
    ) -> float:
        """
        计算预测节点与标准(标签)节点的匹配度，使用 F1 分数。

        对每个预测节点，按 L0 → L1 → L1.5 → L2 顺序尝试匹配
        所有未匹配的标准节点，命中即停止。
        L2 使用 symmetric Jaccard 模式。
        L1.5 / L2 使用固定阈值。

        最终返回:
            precision = 匹配到的预测节点数 / 预测节点总数
            recall    = 匹配到的标准节点数 / 标准节点总数
            F1 = 2 * P * R / (P + R)
        """
        if not pred_nodes or not std_nodes:
            return 0.0

        l1_5_threshold = self.tm_cfg.get("l1_5_threshold", 0.4)
        l2_threshold = self.tm_cfg.get("l2_threshold", 0.4)
        use_l1_5 = self.tm_cfg.get("use_l1_5", True)
        l1_5_context_window = self.tm_cfg.get("l1_5_context_window", 100)

        # 记录已匹配的标准节点索引
        matched_std_indices: set = set()
        matched_pred_count = 0

        for pred_node in pred_nodes:
            pred_name = pred_node["name"]

            for j, std_node in enumerate(std_nodes):
                if j in matched_std_indices:
                    continue

                std_name = std_node["name"]

                # ── L0: 精确子串匹配 (双向) ──
                if _match_level0(pred_name, std_name) or _match_level0(std_name, pred_name):
                    matched_pred_count += 1
                    matched_std_indices.add(j)
                    break

                # ── L1: 归一化子串匹配 (双向) ──
                if _match_level1(pred_name, std_name) or _match_level1(std_name, pred_name):
                    matched_pred_count += 1
                    matched_std_indices.add(j)
                    break

                # ── L1.5: 数值范围 + 上下文 Jaccard ──
                if use_l1_5:
                    ok, _ = _match_level1_5(
                        pred_name, std_name,
                        threshold=l1_5_threshold,
                        mode="symmetric",
                        context_window=l1_5_context_window,
                    )
                    if ok:
                        matched_pred_count += 1
                        matched_std_indices.add(j)
                        break

                # ── L2: symmetric Jaccard ──
                ok, _ = _match_level2(
                    pred_name, std_name,
                    threshold=l2_threshold,
                    mode="symmetric",
                )
                if ok:
                    matched_pred_count += 1
                    matched_std_indices.add(j)
                    break

        precision = matched_pred_count / len(pred_nodes)
        recall = len(matched_std_indices) / len(std_nodes)

        if precision + recall == 0.0:
            return 0.0

        f1 = 2.0 * precision * recall / (precision + recall)
        return f1

    # ================================================================
    # 数据集匹配度
    # ================================================================

    def _compute_dataset_matching(self, pred_nodes: List[Dict]) -> float:
        """
        计算预测节点与标准节点集 node.json 的匹配度。

        策略:
          1. 精确名称命中 → L0_score (默认 1.0)
          2. 否则 → L2 对称 Jaccard 模糊匹配 (动态阈值搜索)
        """
        if not pred_nodes:
            return 0.0

        scores = []
        for node in pred_nodes:
            node_name = node["name"]

            # L0: 精确名称匹配
            if node_name in self._std_nodes_dict:
                scores.append(self.dm_cfg["l0_score"])
                continue

            # L2: 对称 Jaccard 模糊匹配
            fuzzy_score = self._fuzzy_match_against_std_nodes(node_name)
            scores.append(fuzzy_score)

        return sum(scores) / len(scores)

    def _fuzzy_match_against_std_nodes(self, node_name: str) -> float:
        """
        对单个预测节点，在标准节点集中进行 L2 对称 Jaccard 模糊匹配。

        动态阈值策略:
          - 初始阈值 = th (默认 0.5)
          - 每命中一个 (score > current_threshold)，提升阈值为该 score
          - 累计命中 x 次后停止 (默认 5)
          - 返回最终阈值 (即最高命中分); 无命中返回 0.0

        性能优化: 使用预 tokenize 的标准节点 token set 缓存。
        """
        if not self._std_token_list:
            return 0.0

        current_threshold = self.dm_cfg["initial_threshold"]
        max_matches = self.dm_cfg["max_matches_before_stop"]
        match_count = 0

        # 预 tokenize 预测节点名 (只做一次)
        pred_tokens = _tokenize(node_name)
        if not pred_tokens:
            return 0.0
        pred_len = len(pred_tokens)

        for _std_node, std_tokens in self._std_token_list:
            # 快速剪枝: 节点名长度差异过大
            if abs(pred_len - len(std_tokens)) > max(pred_len, len(std_tokens)) * 0.7:
                continue

            # 直接用预计算的 token set 计算 Jaccard
            intersection = len(pred_tokens & std_tokens)
            union = len(pred_tokens | std_tokens)
            score = intersection / union if union > 0 else 0.0

            if score > current_threshold:
                current_threshold = score
                match_count += 1
                if match_count >= max_matches:
                    break

        return current_threshold if match_count > 0 else 0.0

    # ================================================================
    # 主入口
    # ================================================================

    def calculate_reward(
        self,
        completion_hat: str,
        completion: str,
        text: str,
    ) -> Dict[str, Any]:
        """
        计算节点提取任务的综合奖励。

        Args:
            completion_hat: 模型预测的 completion 字符串
            completion:    标准标签 completion 字符串
            text:          输入文本段（prompt 中的用户文本部分）

        Returns:
            {
                "total_reward":           float,   # 最终加权总分
                "structure_score":        float,   # 结构得分
                "length_penalty":         float,   # 长度惩罚 [-1, 0]
                "text_matching_score":    float,   # 文本匹配度 [0, 1]
                "dataset_matching_score": float,   # 数据集匹配度 [0, 1]
                "pred_nodes":             list,    # 预测节点列表
                "pred_count":             int,     # 预测节点数
                "std_count":              int,     # 标准节点数
            }
        """
        result: Dict[str, Any] = {
            "total_reward":           0.0,
            "structure_score":        0.0,
            "length_penalty":         0.0,
            "text_matching_score":    0.0,
            "dataset_matching_score": 0.0,
            "pred_nodes":             [],
            "pred_count":             0,
            "std_count":              0,
        }

        # ── Step 1: 解析预测 completion → 结构得分 ──
        pred_nodes, struct_score = self._parse_completion(completion_hat)
        result["structure_score"] = struct_score
        result["pred_nodes"] = pred_nodes or []
        result["pred_count"] = len(pred_nodes) if pred_nodes else 0

        # 结构得分为 0 → 直接返回
        if struct_score <= 0.0:
            logger.debug("Structure score <= 0, skipping remaining checks")
            return result

        # ── Step 2: 解析标准 completion，获取节点数 ──
        std_nodes, _, std_error = _try_parse_json(
            _extract_json_from_completion(completion) or ""
        )
        if std_error:
            logger.debug("Failed to parse standard completion: %s", std_error)
        num1 = len(std_nodes) if std_nodes else 0
        num2 = len(pred_nodes)
        result["std_count"] = num1

        # ── Step 3: 长度惩罚 ──
        len_penalty = self._compute_length_penalty(num1, num2)
        result["length_penalty"] = len_penalty

        # ── Step 4: 文本匹配度 (预测节点 vs 标准节点, F1 分数) ──
        # 特殊情况：pred 和 std 均为空 → 完美匹配 (模型正确判断无实体)
        if num2 == 0 and num1 == 0:
            text_match = 1.0
        else:
            text_match = self._compute_text_matching(pred_nodes, (std_nodes or []))
        result["text_matching_score"] = text_match

        # ── Step 5: 数据集匹配度 ──
        if num2 == 0 and num1 == 0:
            dataset_match = 1.0
        else:
            dataset_match = self._compute_dataset_matching(pred_nodes)
        result["dataset_matching_score"] = dataset_match

        # ── Step 6: 加权求和 ──
        total = (
            self.weights["structure"]        * struct_score +
            self.weights["length_penalty"]   * len_penalty +
            self.weights["text_matching"]    * text_match +
            self.weights["dataset_matching"] * dataset_match
        )
        result["total_reward"] = total

        return result


# ================================================================
# 便捷函数
# ================================================================

# 模块级单例
_reward_calculator: Optional[NodeRewardCalculator] = None


def get_node_reward_calculator(
    config_path: str = "config/reward.yaml",
) -> NodeRewardCalculator:
    """获取 NodeRewardCalculator 单例。"""
    global _reward_calculator
    if _reward_calculator is None:
        _reward_calculator = NodeRewardCalculator(config_path)
    return _reward_calculator


def compute_node_reward(
    completion_hat: str,
    completion: str,
    text: str,
    config_path: str = "config/reward.yaml",
) -> float:
    """
    便捷函数: 一步计算节点奖励总分。

    Args:
        completion_hat: 模型预测的 completion 字符串
        completion:    标准标签 completion 字符串
        text:          输入文本段
        config_path:   配置文件路径

    Returns:
        总奖励分数 (float)
    """
    calc = get_node_reward_calculator(config_path)
    result = calc.calculate_reward(completion_hat, completion, text)
    return result["total_reward"]
