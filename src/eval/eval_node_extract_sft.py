"""
eval_node_extract_sft.py
========================
Phase 0 节点提取模型 SFT 测试评估器。

加载 QLoRA 微调后的模型，在测试集上评估节点提取性能。
指标包括：结构正确率、精确率、召回率、长度差、类型准确率等。

运行方式:
  python src/eval/eval_node_extract_sft.py
  python src/eval/eval_node_extract_sft.py --config config/sft.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ===================================================================
# 辅助函数
# ===================================================================

def resolve_project_root() -> Path:
    """返回项目根目录。"""
    return Path(__file__).resolve().parent.parent.parent


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """读取 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载 JSONL 文件。"""
    data = []
    logger.info("Loading JSONL: %s", file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info("  loaded %d samples", len(data))
    return data


def load_split_indices(split_path: str) -> Dict[str, List[int]]:
    """加载划分索引文件。"""
    with open(split_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)
    logger.info(
        "Split loaded: train=%d, val=%d, test=%d",
        len(split_info.get("train", [])),
        len(split_info.get("val", [])),
        len(split_info.get("test", [])),
    )
    return split_info


# ===================================================================
# 模型加载 (复用训练时的配置)
# ===================================================================

def _resolve_attn_implementation(use_flash_attn: bool) -> str:
    """按优先级检测可用的 attention 实现: flash_attention_2 → sdpa → eager。"""
    if not use_flash_attn:
        return "eager"
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        logger.warning(
            "Flash Attention 2 enabled but flash_attn not installed. "
            "Falling back to PyTorch SDPA."
        )
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return "sdpa"
    return "eager"


def load_model_for_eval(
    cfg: Dict[str, Any],
    checkpoint_step: Optional[int] = None,
) -> Tuple[Any, AutoTokenizer]:
    """
    加载基座模型 + LoRA 微调权重，返回可用于推理的 (model, tokenizer)。

    推理阶段不需要 prepare_model_for_kbit_training + LoRA 注入，
    直接加载基座模型后包裹 PeftModel。

    Parameters
    ----------
    checkpoint_step : Optional[int]
        若指定，则从 output_dir/checkpoint-{step} 加载最后一层 adapter
        （替代 output_dir 根目录下的最终权重）。
    """
    model_cfg = cfg["model"]

    compute_dtype = getattr(torch, model_cfg["bnb_4bit_compute_dtype"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg["load_in_4bit"],
        bnb_4bit_quant_type=model_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=model_cfg["bnb_4bit_use_double_quant"],
    )

    # ---------- Tokenizer ----------
    logger.info("Loading tokenizer from: %s", model_cfg["base_model_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model_path"],
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- 基座模型 ----------
    attn_implementation = _resolve_attn_implementation(
        model_cfg.get("use_flash_attention_2", False)
    )
    logger.info("Loading base model from: %s (attn: %s, 4-bit)", model_cfg["base_model_path"], attn_implementation)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_path"],
        quantization_config=bnb_config,
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        torch_dtype=compute_dtype,
    )

    # ---------- 确定最后一层 adapter 路径 ----------
    # 检测模式：若 adapter_path 存在 → GRPO 模式（双层 PEFT: 基座 → SFT → GRPO）
    #          否则 → SFT 模式（单层: 基座 → LoRA）
    sft_adapter_path = model_cfg.get("adapter_path", None)
    final_adapter_dir = model_cfg["output_dir"]

    # 若指定了 checkpoint_step，则替换最后一层 adapter 路径为 checkpoint 子目录
    if checkpoint_step is not None:
        checkpoint_dir = os.path.join(final_adapter_dir, f"checkpoint-{checkpoint_step}")
        if not os.path.isdir(checkpoint_dir):
            logger.error(
                "Checkpoint directory not found: %s\n"
                "  output_dir: %s\n"
                "  checkpoint_step: %s",
                checkpoint_dir,
                final_adapter_dir,
                checkpoint_step,
            )
            sys.exit(1)
        logger.info(
            "Using checkpoint (step=%d): %s (overriding output_dir)",
            checkpoint_step,
            checkpoint_dir,
        )
        final_adapter_dir = checkpoint_dir

    if sft_adapter_path is not None:
        # ── GRPO 模式: 基座 → SFT adapter → GRPO adapter ──
        logger.info("Detected GRPO mode (adapter_path is set).")
        logger.info("Loading SFT LoRA adapter from: %s", sft_adapter_path)
        model = PeftModel.from_pretrained(base_model, sft_adapter_path)
        logger.info("Loading GRPO LoRA adapter from: %s", final_adapter_dir)
        model = PeftModel.from_pretrained(model, final_adapter_dir)
    else:
        # ── SFT 模式: 基座 → 单层 adapter ──
        logger.info("Loading LoRA adapter from: %s", final_adapter_dir)
        model = PeftModel.from_pretrained(base_model, final_adapter_dir)

    model.eval()

    return model, tokenizer


# ===================================================================
# Completion 解析
# ===================================================================

# 匹配 ```json ... ``` 代码块（含不完整截断的情况）
_JSON_START_PATTERN = re.compile(r"```json\s*", re.DOTALL)
_JSON_BLOCK_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def _extract_json_content(completion: str) -> Optional[str]:
    """
    从 completion 中提取 JSON 内容，兼容以下情况：
      1. 完整的 ```json ... ``` 块
      2. 生成被截断，缺少闭合 ```（取 ```json 之后到字符串末尾的全部内容）
    """
    # 优先匹配完整块
    m = _JSON_BLOCK_PATTERN.search(completion)
    if m:
        return m.group(1).strip()

    # 回退：只有开头 ```json，没有闭合 ```
    m_start = _JSON_START_PATTERN.search(completion)
    if m_start:
        # 取 ```json 之后到字符串末尾
        raw = completion[m_start.end():]
        # 去掉末尾可能残留的不完整 ```（如 `` 或 `）
        raw = re.sub(r"\s*`{0,2}\s*$", "", raw)
        if raw.strip():
            return raw.strip()

    return None


def _try_repair_truncated_json(json_str: str) -> Optional[str]:
    """
    尝试修复被截断的 JSON：
      - 末尾不完整字符串值 → 用空字符串闭合
      - 缺少 ] 或 } → 补充闭合
    返回修复后的 JSON 字符串，无法修复则返回 None。
    """
    if not json_str:
        return None

    s = json_str.strip()

    # 统计括号
    open_braces = s.count("{") - s.count("}")
    open_brackets = s.count("[") - s.count("]")

    # 如果末尾是不完整的字符串值（如 "type），尝试用 " 闭合
    # 检测：最后一个非空白字符是否在未闭合的字符串中
    in_string = False
    escaped = False
    for ch in s:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string

    # 如果在字符串中间被截断，先闭合字符串
    if in_string:
        s += '"'

    # 重新计算（因为可能添加了引号后改变了括号计数）
    open_braces = s.count("{") - s.count("}")
    open_brackets = s.count("[") - s.count("]")

    # 闭合括号
    if open_brackets > 0:
        s += "]" * open_brackets
    if open_braces > 0:
        s += "}" * open_braces

    return s


def parse_completion_to_nodes(completion: str) -> Optional[List[Dict[str, str]]]:
    """
    将模型输出的 completion 字符串解析为节点列表。

    容错处理：
      - 若 ```json ... ``` 不完整（生成被截断），尝试提取并修复 JSON
      - 内部 JSON 可解析
      - 顶层有 "nodes" 键，值为数组
      - 每个元素必须仅有 "name" 和 "type" 两个键

    Returns
    -------
    Optional[List[Dict[str, str]]]
        [{"name": ..., "type": ...}, ...]  解析成功
        None                               解析失败
    """
    if not completion or not isinstance(completion, str):
        return None

    # 提取 JSON 内容（含截断容错）
    json_str = _extract_json_content(completion)
    if not json_str:
        return None

    # 尝试直接解析
    obj = None
    try:
        obj = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        # 尝试修复截断的 JSON
        repaired = _try_repair_truncated_json(json_str)
        if repaired:
            try:
                obj = json.loads(repaired)
            except (json.JSONDecodeError, ValueError):
                pass

    if obj is None:
        return None

    # 按格式校验
    if not isinstance(obj, dict):
        return None
    if "nodes" not in obj:
        return None
    nodes = obj["nodes"]
    if not isinstance(nodes, list):
        return None

    # 每个节点必须仅有 name 和 type 两个键
    result = []
    for node in nodes:
        if not isinstance(node, dict):
            return None
        keys = set(node.keys())
        if keys != {"name", "type"}:
            return None
        result.append({
            "name": str(node["name"]),
            "type": str(node["type"]),
        })

    return result


# ===================================================================
# 逐样本指标计算
# ===================================================================

def compute_sample_metrics(
    pred_nodes: List[Dict[str, str]],
    gt_nodes: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    对单个样本计算所有指标。

    Returns
    -------
    dict with keys:
        precision, recall, length_diff, correct_count,
        pred_count, gt_count, correct_type_count
    """
    # 构造 (name, type) 元组集合
    pred_set = set()
    for n in pred_nodes:
        pred_set.add((n["name"], n["type"]))

    gt_set = set()
    for n in gt_nodes:
        gt_set.add((n["name"], n["type"]))

    # 交集
    intersection = pred_set & gt_set
    correct_count = len(intersection)
    pred_count = len(pred_nodes)
    gt_count = len(gt_nodes)

    # 精确率 = TP / 预测数
    precision = correct_count / pred_count if pred_count > 0 else 0.0
    # 召回率 = TP / 标签数
    recall = correct_count / gt_count if gt_count > 0 else 0.0
    # 长度差
    length_diff = abs(pred_count - gt_count)

    # 正确类型数: 对每个 (name, type) 在交集中的节点，type 匹配 (已经蕴含在 set 交集里)
    # type 正确 ≡ name & type 均匹配 → 就是 intersection 的大小
    correct_type_count = correct_count

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "length_diff": length_diff,
        "correct_count": correct_count,
        "pred_count": pred_count,
        "gt_count": gt_count,
        "correct_type_count": correct_type_count,
    }


# ===================================================================
# 聚合指标计算
# ===================================================================

def compute_aggregate_metrics(
    details: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    从逐样本详情列表计算聚合指标。

    `details` 中每个元素应包含:
      - parse_success: bool
      - metrics: dict  (仅 parse_success=True 的样本有此字段)

    Returns
    -------
    dict 聚合指标
    """
    total = len(details)
    success_details = [d for d in details if d.get("parse_success", False)]
    success_count = len(success_details)

    # 结构正确率
    struct_accuracy = success_count / total if total > 0 else 0.0

    # 仅对可解析样本统计
    if success_count == 0:
        return {
            "total_samples": total,
            "parse_success_count": success_count,
            "struct_accuracy": round(struct_accuracy, 4),
            "per_sample_precision_mean": 0.0,
            "per_sample_precision_std": 0.0,
            "per_sample_recall_mean": 0.0,
            "per_sample_recall_std": 0.0,
            "per_sample_length_diff_mean": 0.0,
            "per_sample_length_diff_std": 0.0,
            "overall_precision": 0.0,
            "overall_recall": 0.0,
            "type_accuracy": 0.0,
        }

    # 提取指标数组
    precisions = np.array([d["metrics"]["precision"] for d in success_details])
    recalls = np.array([d["metrics"]["recall"] for d in success_details])
    length_diffs = np.array([d["metrics"]["length_diff"] for d in success_details], dtype=np.float64)

    # 累计值
    sum_correct = sum(d["metrics"]["correct_count"] for d in success_details)
    sum_pred = sum(d["metrics"]["pred_count"] for d in success_details)
    sum_gt = sum(d["metrics"]["gt_count"] for d in success_details)
    sum_correct_type = sum(d["metrics"]["correct_type_count"] for d in success_details)

    overall_precision = sum_correct / sum_pred if sum_pred > 0 else 0.0
    overall_recall = sum_correct / sum_gt if sum_gt > 0 else 0.0
    type_accuracy = sum_correct_type / sum_correct if sum_correct > 0 else 0.0

    return {
        "total_samples": total,
        "parse_success_count": success_count,
        "struct_accuracy": round(struct_accuracy, 4),
        "per_sample_precision_mean": round(float(np.mean(precisions)), 4),
        "per_sample_precision_std": round(float(np.std(precisions, ddof=1)) if success_count > 1 else 0.0, 4),
        "per_sample_recall_mean": round(float(np.mean(recalls)), 4),
        "per_sample_recall_std": round(float(np.std(recalls, ddof=1)) if success_count > 1 else 0.0, 4),
        "per_sample_length_diff_mean": round(float(np.mean(length_diffs)), 4),
        "per_sample_length_diff_std": round(float(np.std(length_diffs, ddof=1)) if success_count > 1 else 0.0, 4),
        "overall_precision": round(overall_precision, 4),
        "overall_recall": round(overall_recall, 4),
        "type_accuracy": round(type_accuracy, 4),
    }


# ===================================================================
# 主评估入口
# ===================================================================

def run_evaluation(
    config_path: str = "config/sft.yaml",
    max_samples: Optional[int] = None,
    checkpoint_step: Optional[int] = None,
) -> None:
    """执行 SFT 模型测试评估。"""
    # ---------- 加载配置 ----------
    cfg = load_yaml_config(Path(config_path))
    logger.info("Configuration loaded from: %s", config_path)

    # ---------- 加载模型 ----------
    model, tokenizer = load_model_for_eval(cfg, checkpoint_step=checkpoint_step)

    # ---------- 加载测试集 ----------
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})
    all_data = load_jsonl(data_cfg["train_data_path"])
    split_indices = load_split_indices(data_cfg["split_info_path"])
    test_indices = split_indices.get("test", [])

    if not test_indices:
        logger.error("No test indices found in split file.")
        sys.exit(1)

    # 可选截断：仅评估前 N 个样本
    if max_samples is not None and max_samples > 0:
        test_indices = test_indices[:max_samples]
        logger.info("Limiting to first %d test samples for quick evaluation.", len(test_indices))

    # ---------- 逐样本评估 ----------
    details: List[Dict[str, Any]] = []
    raw_outputs: List[Dict[str, str]] = []
    disable_sampling = os.environ.get("EVAL_DISABLE_SAMPLING", "0") == "1"

    logger.info("Starting evaluation on %d test samples...", len(test_indices))

    for idx in tqdm(test_indices, desc="Evaluating", unit="sample"):
        if idx >= len(all_data):
            logger.warning("Index %d out of range (%d), skipping.", idx, len(all_data))
            continue

        sample = all_data[idx]
        prompt = sample["prompt"]
        gt_completion_str = sample["completion"]

        # 解析标签 completion → 节点列表
        gt_nodes = parse_completion_to_nodes(gt_completion_str)
        if gt_nodes is None:
            # 标签无法解析时跳过该样本
            logger.warning("Sample %d: ground-truth completion parse failed, skipping.", idx)
            continue

        # 模型推理
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            if disable_sampling:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        # 截取生成的 completion（去掉 prompt 部分）
        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_len:]
        pred_completion_str = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 保存模型原始输出
        raw_outputs.append({
            "prompt": sample.get("prompt", ""),
            "completion": pred_completion_str,
            "text": sample.get("text", ""),
        })

        # 解析预测 completion
        pred_nodes = parse_completion_to_nodes(pred_completion_str)

        if pred_nodes is None:
            details.append({
                "sample_idx": idx,
                "parse_success": False,
                "pred_completion_raw": pred_completion_str,
                "gt_completion_raw": gt_completion_str,
                "gt_nodes": gt_nodes,
            })
            continue

        # 计算指标
        metrics = compute_sample_metrics(pred_nodes, gt_nodes)

        details.append({
            "sample_idx": idx,
            "parse_success": True,
            "metrics": metrics,
            "pred_nodes": pred_nodes,
            "gt_nodes": gt_nodes,
            "pred_completion_raw": pred_completion_str,
            "gt_completion_raw": gt_completion_str,
        })

    # ---------- 聚合计算 ----------
    aggregate = compute_aggregate_metrics(details)

    # ---------- 保存结果 ----------
    # 根据配置自动检测模式，使用对应的默认输出路径
    _is_grpo = cfg["model"].get("adapter_path") is not None
    _default_out_dir = "/workspace/Output/GRPODatasets" if _is_grpo else "/workspace/Output/SFTDatasets"
    _prefix = "grpo" if _is_grpo else "sft"

    details_output_path = eval_cfg.get(
        "test_details_path",
        f"{_default_out_dir}/{_prefix}_node_test_details.json",
    )
    aggregate_output_path = eval_cfg.get(
        "test_output_path",
        f"{_default_out_dir}/{_prefix}_node_test_output.json",
    )
    raw_output_path = eval_cfg.get(
        "test_raw_output_path",
        f"{_default_out_dir}/{_prefix}_node_test_raw.json",
    )

    os.makedirs(os.path.dirname(details_output_path), exist_ok=True)
    with open(details_output_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    logger.info("Test details saved to: %s", details_output_path)

    os.makedirs(os.path.dirname(aggregate_output_path), exist_ok=True)
    with open(aggregate_output_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    logger.info("Aggregate metrics saved to: %s", aggregate_output_path)

    # 保存模型原始输出 (JSONL 格式, 与 sft_node_train.json 类似)
    os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
    with open(raw_output_path, "w", encoding="utf-8") as f:
        for item in raw_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Raw model outputs saved to: %s (%d samples)", raw_output_path, len(raw_outputs))

    # ---------- 打印摘要 ----------
    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    for key, value in aggregate.items():
        logger.info("  %-35s: %s", key, value)
    logger.info("=" * 60)


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="SFT Node Extraction Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="config/sft.yaml",
        help="Path to sft.yaml config file (default: config/sft.yaml)",
    )
    parser.add_argument(
        "--disable-sampling",
        action="store_true",
        help="Use greedy decoding instead of sampling.",
    )
    parser.add_argument(
        "-n",
        "--max-samples",
        type=int,
        default=None,
        help="Only evaluate the first N test samples (for quick testing).",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        metavar="STEP",
        help="Load the last-layer adapter from output_dir/checkpoint-STEP "
             "instead of output_dir itself.",
    )
    args = parser.parse_args()

    if args.disable_sampling:
        os.environ["EVAL_DISABLE_SAMPLING"] = "1"

    config_path = resolve_project_root() / args.config
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    run_evaluation(
        str(config_path),
        max_samples=args.max_samples,
        checkpoint_step=args.checkpoint,
    )


if __name__ == "__main__":
    main()
