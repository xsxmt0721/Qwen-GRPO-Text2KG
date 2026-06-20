"""
nodes_grpotrainer.py
====================
GRPO 节点提取模型训练器。

基于 TRL GRPOTrainer 对 Qwen2.5-1.5B-Instruct + SFT LoRA 进行 GRPO 强化微调。

流程:
  1. 加载基座模型 (INT4 QLoRA) + SFT LoRA 适配器
  2. 注入新的 LoRA 层作为可训练参数
  3. 构建 GRPOTrainer (TRL)，使用 NodeRewardCalculator 作为奖励函数
  4. 在 GRPO 数据集上训练，监控多项指标

运行方式:
  python src/alignment/nodes_grpotrainer.py --config config/grpo_nodes.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ── TRL imports (容器内可用) ──
try:
    from trl import GRPOTrainer, GRPOConfig
    _TRL_AVAILABLE = True
except ImportError:
    _TRL_AVAILABLE = False

# ---------------------------------------------------------------------------
# 项目路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.nodes_reward import NodeRewardCalculator

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ===================================================================
# 辅助: 注意力实现检测
# ===================================================================

def _resolve_attn_implementation(use_flash_attn: bool) -> str:
    """按优先级检测: flash_attention_2 → sdpa → eager。"""
    if not use_flash_attn:
        return "eager"
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        logger.warning(
            "Flash Attention 2 not installed. Falling back to PyTorch SDPA."
        )
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return "sdpa"
    return "eager"


# ===================================================================
# 配置加载
# ===================================================================

def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_split_indices(split_path: str) -> Dict[str, List[int]]:
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===================================================================
# 模型加载
# ===================================================================

def load_model_for_grpo(cfg: Dict[str, Any]) -> Tuple[Any, AutoTokenizer]:
    """
    加载基座模型 + SFT LoRA 适配器, 并注入新的可训练 LoRA 层。

    GRPOTrainer 会自动保存初始模型状态作为 ref_model，无需手动构建。

    Returns:
        (model, tokenizer)
        - model:      带 SFT 适配器 + 新 LoRA 的可训练策略模型
        - tokenizer
    """
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]

    use_4bit = model_cfg.get("load_in_4bit", False)
    if use_4bit:
        compute_dtype = getattr(torch, model_cfg["bnb_4bit_compute_dtype"])
    else:
        compute_dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))

    # ── Tokenizer ──
    logger.info("Loading tokenizer from: %s", model_cfg["base_model_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model_path"],
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_implementation = _resolve_attn_implementation(
        model_cfg.get("use_flash_attention_2", False)
    )

    # ── 加载基座模型 ──
    load_mode = "4-bit" if use_4bit else "full BF16"
    logger.info(
        "Loading base model from: %s (attn: %s, %s)",
        model_cfg["base_model_path"], attn_implementation, load_mode,
    )

    load_kwargs: Dict[str, Any] = {
        "device_map": model_cfg.get("device_map", "auto"),
        "trust_remote_code": True,
        "attn_implementation": attn_implementation,
        "torch_dtype": compute_dtype,
    }
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_cfg["bnb_4bit_use_double_quant"],
        )
        load_kwargs["quantization_config"] = bnb_config

    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_path"],
        **load_kwargs,
    )

    # ── 加载 SFT LoRA → 策略模型 ──
    adapter_path = model_cfg["adapter_path"]
    logger.info("Loading SFT LoRA adapter from: %s (4-bit=%s)", adapter_path, use_4bit)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # ── 注入新的 GRPO LoRA ──
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.train()
    model.print_trainable_parameters()

    # NOTE: GRPOTrainer 内部会自动将当前模型权重保存为 ref_model，无需手动构建
    logger.info("Model setup complete (ref_model will be auto-created by GRPOTrainer).")
    return model, tokenizer


# ===================================================================
# 奖励函数封装
# ===================================================================

class GRPORewardWrapper:
    """
    将 NodeRewardCalculator 封装为 GRPOTrainer 所需的批量奖励函数。

    GRPOTrainer 调用签名: reward_func(completions, prompts, **dataset_columns)
    通过 dataset_columns 传入 ground-truth completion 和 text，按 batch 内索引对齐。
    """

    # TRL 要求 reward function 有 __name__ 属性
    __name__ = "node_reward"

    def __init__(
        self,
        reward_config_path: str = "config/reward.yaml",
    ):
        self._calc = NodeRewardCalculator(reward_config_path)
        self._call_count = 0
        self._total_time = 0.0

    def __call__(
        self,
        completions: List[str],
        prompts: List[str] = None,
        **dataset_columns,
    ) -> List[float]:
        """
        Args:
            completions:    模型生成的 completion_hat 列表
            prompts:        prompt 列表 (TRL 传入)
            dataset_columns: 数据集附加列，包含:
                - gt_completion: List[str]  ground-truth completion 列表
                - text:          List[str]  输入文本列表

        Returns:
            与 completions 等长的奖励列表
        """
        import time as _time
        _t0 = _time.time()

        # 从 dataset_columns 获取 ground truth，按 batch 内索引对齐
        # 注意: TRL v0.15.2 会过滤掉 "completion" 键，因此数据集中使用 "gt_completion"
        gt_completions = dataset_columns.get("gt_completion", [])
        gt_texts = dataset_columns.get("text", [])

        rewards = []
        for i, completion_hat in enumerate(completions):
            gt_comp = gt_completions[i] if i < len(gt_completions) else ""
            text = gt_texts[i] if i < len(gt_texts) else ""

            result = self._calc.calculate_reward(completion_hat, gt_comp, text)
            rewards.append(result["total_reward"])

            # DEBUG: 打印前 3 个 batch 的预测 vs 标准答案，验证映射表是否生效
            if self._call_count < 3:
                logger.warning(
                    "DEBUG batch#%d sample#%d | "
                    "completion_hat: %s | "
                    "gt_completion: %s",
                    self._call_count, i,
                    completion_hat[:120].replace("\n", "\\n") if completion_hat else "<EMPTY>",
                    gt_comp[:120].replace("\n", "\\n") if gt_comp else "<EMPTY>",
                )

        _elapsed = _time.time() - _t0
        self._call_count += 1
        self._total_time += _elapsed

        return rewards


# ===================================================================
# 数据集构建
# ===================================================================

def build_grpo_dataset(
    data_path: str,
    split_path: str,
    split_key: str,
    max_samples: Optional[int] = None,
) -> HFDataset:
    """
    从 JSONL 和 split 索引构建 HuggingFace Dataset。

    保留列: prompt, gt_completion, text
    (注意: 列名使用 gt_completion 而非 completion, 因为 TRL v0.15.2 在传给
     reward_func 的 **reward_kwargs 中会显式过滤掉 "completion" 键)
    """
    all_data = load_jsonl(data_path)
    split_info = load_split_indices(split_path)
    indices = split_info.get(split_key, [])

    if max_samples is not None and max_samples > 0:
        indices = indices[:max_samples]

    samples = []
    for idx in indices:
        if idx >= len(all_data):
            continue
        item = all_data[idx]
        samples.append({
            "prompt": item["prompt"],
            "gt_completion": item["completion"],
            "text": item.get("text", ""),
        })

    dataset = HFDataset.from_list(samples)
    logger.info(
        "Built %s dataset: %d samples (from %s)", split_key, len(dataset), data_path
    )
    return dataset


# ===================================================================
# 主训练入口
# ===================================================================

def run_grpo_training(
    config_path: str = "config/grpo_nodes.yaml",
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> None:
    if not _TRL_AVAILABLE:
        raise ImportError("TRL library is required for GRPO training. "
                          "Install with: pip install trl")

    # ── 加载配置 ──
    config_full_path = PROJECT_ROOT / config_path
    cfg = load_yaml_config(config_full_path)
    logger.info("Configuration loaded from: %s", config_path)

    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    log_cfg = cfg["logging"]
    reward_cfg_path = cfg["reward"]["config_path"]

    # ── 加载模型 ──
    model, tokenizer = load_model_for_grpo(cfg)

    # ── 构建数据集 ──
    train_dataset = build_grpo_dataset(
        data_cfg["train_data_path"],
        data_cfg["split_info_path"],
        "train",
        max_samples=max_train_samples,
    )
    eval_dataset = build_grpo_dataset(
        data_cfg.get("eval_data_path", data_cfg["train_data_path"]),
        data_cfg["split_info_path"],
        "val",
        max_samples=max_eval_samples,
    )

    # ── 包装奖励函数 ──
    # reward_func 通过 **dataset_columns 按索引直接获取 gt_completion 和 text。
    # 注意: 数据集中列名为 gt_completion (非 completion), 因为 TRL v0.15.2 会过滤掉
    # "completion" 键。与 test_node_reward.py 的索引对齐逻辑一致。
    reward_func = GRPORewardWrapper(reward_cfg_path)

    # ── GRPO 训练参数 ──
    grpo_config = GRPOConfig(
        # 基础训练参数
        output_dir=cfg["model"]["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        optim=train_cfg["optim"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=train_cfg["bf16"],
        fp16=train_cfg["fp16"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        seed=train_cfg["seed"],
        remove_unused_columns=train_cfg.get("remove_unused_columns", False),

        # GRPO 特有参数
        beta=train_cfg.get("kl_beta", 0.04),
        num_generations=train_cfg["num_generations"],
        temperature=train_cfg["temperature"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        max_prompt_length=train_cfg.get("max_prompt_length", 1536),
        max_completion_length=train_cfg.get("max_completion_length", 512),

        # 日志 / 保存
        logging_steps=train_cfg["logging_steps"],
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        save_strategy=train_cfg["save_strategy"],
        eval_strategy=train_cfg["eval_strategy"],
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_reward_mean"),

        # Logging
        logging_dir=log_cfg["log_dir"],
        report_to=["tensorboard"] if log_cfg.get("use_tensorboard", True) else [],
    )

    # ── 创建 Trainer ──
    # 对于 double-PEFT (SFT adapter + GRPO LoRA)，显式指定 ref_model
    # 避免 GRPOTrainer 尝试 deep-copy 整个 4-bit 量化模型导致极慢
    trainer_kwargs = dict(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=[reward_func],
        processing_class=tokenizer,
    )
    # ref_model: 若 TRL 版本支持，传入 None 让其用 disable_adapter 自动处理;
    # 若不支持且导致拷贝极慢，可传入 model (但冻结 GRPO LoRA 前向)
    try:
        trainer = GRPOTrainer(**trainer_kwargs)
    except TypeError:
        # 旧版 TRL 可能不接受某些参数
        trainer_kwargs.pop("processing_class", None)
        trainer = GRPOTrainer(**trainer_kwargs)

    # ── 训练 ──
    logger.info("=" * 60)
    logger.info("Starting GRPO training")
    logger.info("=" * 60)
    logger.info("  Model:            %s", cfg["model"]["base_model_path"])
    logger.info("  SFT adapter:      %s", cfg["model"]["adapter_path"])
    logger.info("  Output dir:       %s", cfg["model"]["output_dir"])
    logger.info("  Train samples:    %d", len(train_dataset))
    logger.info("  Eval samples:     %d", len(eval_dataset))
    logger.info("  Num generations:  %d", train_cfg["num_generations"])
    logger.info("  Temperature:      %.2f", train_cfg["temperature"])
    logger.info("  Learning rate:    %.1e", train_cfg["learning_rate"])
    logger.info("  Log dir:          %s", log_cfg["log_dir"])
    logger.info("=" * 60)

    # ── 训练前计时诊断 ──
    import time as _time
    _t_train_start = _time.time()
    logger.info("[TIMING] Starting trainer.train() at t=0s ...")

    trainer.train()

    _t_train_end = _time.time()
    logger.info("[TIMING] trainer.train() completed in %.1fs (%.1f min)",
                _t_train_end - _t_train_start,
                (_t_train_end - _t_train_start) / 60.0)

    # ── 保存最终模型 ──
    final_output = cfg["model"]["output_dir"]
    logger.info("Saving final model to: %s", final_output)
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)

    logger.info("GRPO training completed!")


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO Node Extraction Training")
    parser.add_argument(
        "--config", type=str, default="config/grpo_nodes.yaml",
        help="Path to GRPO config file (default: config/grpo_nodes.yaml)",
    )
    parser.add_argument(
        "-n", "--max-train-samples", type=int, default=None,
        help="Limit training samples (for quick testing).",
    )
    parser.add_argument(
        "--max-eval-samples", type=int, default=None,
        help="Limit eval samples (for quick testing).",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    run_grpo_training(
        config_path=str(config_path),
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )


if __name__ == "__main__":
    main()
