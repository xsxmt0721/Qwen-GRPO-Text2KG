"""
nodes_extract_sft_trainer.py
============================
Phase 0 节点提取模型 SFT 监督微调训练器。

利用 TRL SFTTrainer + QLoRA 对 Qwen2.5-1.5B-Instruct 进行监督微调。
- INT4 量化加载 (bitsandbytes)
- BF16 混合精度训练
- Flash Attention 2 加速
- DataCollatorForCompletionOnlyLM: 仅对 assistant 回复部分计算 loss
- TensorBoard 监控: train/eval loss, PPL, L2 grad norm, lr, tokens/s, GPU 显存

运行方式:
  python src/alignment/nodes_extract_sft_trainer.py
  python src/alignment/nodes_extract_sft_trainer.py --config config/sft.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_attn_implementation(use_flash_attn: bool) -> str:
    """
    按优先级检测可用的 attention 实现：
      flash_attention_2 → sdpa → eager

    Returns
    -------
    str
        实际使用的 attn_implementation 名称。
    """
    if not use_flash_attn:
        logger.info("Flash Attention 2 disabled in config, using 'eager' attention.")
        return "eager"

    # 尝试 flash_attention_2
    try:
        import flash_attn  # noqa: F401
        logger.info("Flash Attention 2 detected, using 'flash_attention_2'.")
        return "flash_attention_2"
    except ImportError:
        logger.warning(
            "Flash Attention 2 is enabled in config but flash_attn package is not installed. "
            "Falling back to PyTorch SDPA (scaled_dot_product_attention)."
        )

    # 尝试 PyTorch 内置 SDPA (PyTorch >= 2.0)
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        logger.info("PyTorch SDPA available, using 'sdpa' attention.")
        return "sdpa"

    logger.warning("Neither flash_attn nor SDPA available, using 'eager' attention (slow).")
    return "eager"

def resolve_project_root() -> Path:
    """返回项目根目录。"""
    return Path(__file__).resolve().parent.parent.parent


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """读取 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载 JSONL 文件，返回列表。"""
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
# 自定义 Metrics Callback
# ===================================================================

class SFTMetricsCallback(TrainerCallback):
    """
    自定义回调，记录额外训练指标到 TensorBoard:
      - eval_PPL (困惑度)
      - tokens_per_second (吞吐量)
      - gpu_memory_allocated_gb (GPU 显存占用)
    """

    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self._train_start_time: Optional[float] = None
        self._train_step_times: List[float] = []
        self._last_log_time: Optional[float] = None
        self._total_observed_tokens: int = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._train_start_time = time.time()
        self._last_log_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            logger.info("GPU peak memory stats reset at training start.")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """每个 step 结束时记录 GPU 显存。"""
        if state.global_step % max(args.logging_steps, 1) == 0 and torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            gpu_mem_gb_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            current_step = state.global_step
            self.writer.add_scalar("system/gpu_memory_allocated_gb", gpu_mem_gb, current_step)
            self.writer.add_scalar("system/gpu_memory_reserved_gb", gpu_mem_gb_reserved, current_step)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """每次日志时记录 PPL 和 tokens/s。"""
        if logs is None:
            return

        current_step = state.global_step

        # 验证集困惑度 PPL = exp(eval_loss)
        if "eval_loss" in logs:
            eval_loss = logs["eval_loss"]
            eval_ppl = math.exp(eval_loss) if eval_loss < 100 else float("inf")
            self.writer.add_scalar("eval/perplexity", eval_ppl, current_step)
            logger.info("  eval PPL: %.4f", eval_ppl)

        # 每秒 tokens 吞吐量 (基于 total_flos 估算)
        if "total_flos" in logs and self._last_log_time is not None:
            now = time.time()
            elapsed = now - self._last_log_time
            self._last_log_time = now
            if elapsed > 0 and hasattr(state, "num_input_tokens_seen"):
                # TRL SFTTrainer 可能不提供此属性，回退到基于 step 估算
                pass

        # 记录训练集 loss
        if "loss" in logs:
            self.writer.add_scalar("train/loss", logs["loss"], current_step)

        # 记录当前学习率
        if "learning_rate" in logs:
            self.writer.add_scalar("train/learning_rate", logs["learning_rate"], current_step)

        # 记录 L2 梯度范数
        if "grad_norm" in logs:
            self.writer.add_scalar("train/grad_norm", logs["grad_norm"], current_step)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """每次评估结束后额外记录。"""
        if metrics is None:
            return
        current_step = state.global_step
        if "eval_loss" in metrics:
            ppl = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 100 else float("inf")
            self.writer.add_scalar("eval/perplexity", ppl, current_step)


# ===================================================================
# 数据集构建 & 手动 tokenization（prompt 部分 label 设为 -100）
# ===================================================================

def build_sft_datasets(
    data_path: str,
    split_indices: Dict[str, List[int]],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> Dict[str, HFDataset]:
    """
    加载原始数据并按划分构建已 tokenize 的 train / val 数据集。

    prompt 的 token 对应 label 设为 -100（不计算 loss），
    仅 completion 部分参与 loss 计算。
    严格不包含测试集。
    """
    all_data = load_jsonl(data_path)

    datasets: Dict[str, HFDataset] = {}

    def _tokenize_fn(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """批量 tokenize：分别编码 prompt 和 completion，拼接并设置 labels。"""
        prompts = examples["prompt"]
        completions = examples["completion"]

        # 分别 tokenize（不添加特殊 token，prompt 中已含 <|im_start|>/<|im_end|>）
        prompt_enc = tokenizer(
            prompts,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )
        completion_enc = tokenizer(
            completions,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )

        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for p_ids, c_ids, p_mask, c_mask in zip(
            prompt_enc["input_ids"],
            completion_enc["input_ids"],
            prompt_enc["attention_mask"],
            completion_enc["attention_mask"],
        ):
            # 拼接 prompt + completion
            input_ids = p_ids + c_ids
            attention_mask = p_mask + c_mask
            # prompt 部分 label = -100，仅 completion 计算 loss
            labels = [-100] * len(p_ids) + c_ids

            # 截断到 max_seq_length
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]
                attention_mask = attention_mask[:max_seq_length]
                labels = labels[:max_seq_length]

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }

    for split_name in ("train", "val"):
        indices = split_indices.get(split_name, [])
        if not indices:
            logger.warning("Split '%s' has no indices, skipping.", split_name)
            continue

        subset = [
            {"prompt": all_data[i]["prompt"], "completion": all_data[i]["completion"]}
            for i in indices
            if i < len(all_data)
        ]
        hf_ds = HFDataset.from_list(subset)
        # 批量 tokenize
        hf_ds = hf_ds.map(
            _tokenize_fn,
            batched=True,
            batch_size=64,
            remove_columns=hf_ds.column_names,
        )
        datasets[split_name] = hf_ds
        logger.info("  %s dataset: %d samples", split_name, len(hf_ds))

    return datasets


# ===================================================================
# 模型加载
# ===================================================================

def build_model_and_tokenizer(cfg: Dict[str, Any]) -> tuple:
    """
    加载 Qwen2.5 基座模型 (INT4 量化) 和 tokenizer。

    Returns
    -------
    (model, tokenizer)
    """
    model_cfg = cfg["model"]

    # ---------- 量化配置 ----------
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

    # Qwen2.5 使用 <|im_start|> / <|im_end|> 特殊 token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # SFT 训练使用 right padding

    # ---------- 模型 ----------
    attn_implementation = _resolve_attn_implementation(
        model_cfg.get("use_flash_attention_2", False)
    )
    logger.info("Loading model from: %s (attn: %s, 4-bit)", model_cfg["base_model_path"], attn_implementation)

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_path"],
        quantization_config=bnb_config,
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        torch_dtype=compute_dtype,
    )

    # QLoRA 预处理: 为 k-bit 训练做准备
    model = prepare_model_for_kbit_training(model)

    # 梯度检查点
    if model_cfg.get("use_gradient_checkpointing", False):
        gckpt_kwargs = model_cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False})
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gckpt_kwargs)
        logger.info("Gradient checkpointing enabled.")

    # ---------- LoRA ----------
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ===================================================================
# 主训练入口
# ===================================================================

def run_sft_training(config_path: str = "config/sft.yaml") -> None:
    """执行 SFT 训练主流程。"""
    # ---------- 加载配置 ----------
    cfg = load_yaml_config(Path(config_path))
    logger.info("Configuration loaded from: %s", config_path)

    # ---------- 加载模型 ----------
    model, tokenizer = build_model_and_tokenizer(cfg)
    model.config.use_cache = False  # 训练时关闭 cache

    # ---------- 提取训练 & 日志配置 ----------
    train_cfg = cfg["training"]
    log_cfg = cfg["logging"]

    # ---------- 准备数据集 (tokenize with label masking) ----------
    data_cfg = cfg["data"]
    tokenizer.model_max_length = train_cfg["max_seq_length"]
    split_indices = load_split_indices(data_cfg["split_info_path"])
    datasets = build_sft_datasets(
        data_path=data_cfg["train_data_path"],
        split_indices=split_indices,
        tokenizer=tokenizer,
        max_seq_length=train_cfg["max_seq_length"],
    )

    train_dataset = datasets.get("train")
    eval_dataset = datasets.get("val")

    if train_dataset is None:
        raise ValueError("Train dataset is empty. Check split indices and data path.")

    # ---------- 训练参数 ----------
    training_args = TrainingArguments(
        # 输出 & 日志
        output_dir=cfg["model"]["output_dir"],
        logging_dir=log_cfg["log_dir"],
        report_to=["tensorboard"] if log_cfg.get("use_tensorboard", True) else [],

        # 精度
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),

        # 训练超参
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),

        # 策略
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_steps=train_cfg.get("eval_steps", 200),
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_cfg.get("greater_is_better", False),

        # 数据
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        remove_unused_columns=train_cfg.get("remove_unused_columns", False),
        seed=train_cfg.get("seed", 42),

        # 其他
        run_name="sft_text2node",
    )

    # ---------- TensorBoard Writer (手动管理额外指标) ----------
    tb_writer = SummaryWriter(log_dir=log_cfg["log_dir"])

    # ---------- Data Collator (自定义: labels 用 -100 padding) ----------
    class CustomDataCollator:
        """
        自定义 collator：input_ids/attention_mask 用 pad_token_id 填充，
        labels 用 -100 填充（loss 忽略位）。
        使用纯 PyTorch padding 避免 tokenizer 的 warn 日志。
        """
        def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: int = 8):
            self.pad_token_id = tokenizer.pad_token_id
            self.pad_to_multiple_of = pad_to_multiple_of

        def _pad_tensor(self, values: List[List[int]], pad_value: int) -> torch.Tensor:
            max_len = max(len(v) for v in values)
            if self.pad_to_multiple_of > 1:
                max_len = ((max_len + self.pad_to_multiple_of - 1)
                           // self.pad_to_multiple_of) * self.pad_to_multiple_of
            padded = torch.full((len(values), max_len), pad_value, dtype=torch.long)
            for i, v in enumerate(values):
                padded[i, :len(v)] = torch.tensor(v, dtype=torch.long)
            return padded

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            return {
                "input_ids": self._pad_tensor([f["input_ids"] for f in features], self.pad_token_id),
                "attention_mask": self._pad_tensor([f["attention_mask"] for f in features], 0),
                "labels": self._pad_tensor([f["labels"] for f in features], -100),
            }

    collator = CustomDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    # ---------- Trainer (standard HuggingFace, with pre-masked labels) ----------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[SFTMetricsCallback(writer=tb_writer)],
    )

    # ---------- 开始训练 ----------
    logger.info("Starting SFT training...")
    trainer.train()

    # ---------- 保存最终模型 ----------
    logger.info("Saving final model to: %s", cfg["model"]["output_dir"])
    trainer.save_model(cfg["model"]["output_dir"])
    tokenizer.save_pretrained(cfg["model"]["output_dir"])

    tb_writer.close()
    logger.info("Training completed.")


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="SFT Node Extraction Trainer")
    parser.add_argument(
        "--config",
        type=str,
        default="config/sft.yaml",
        help="Path to sft.yaml config file (default: config/sft.yaml)",
    )
    args = parser.parse_args()

    config_path = resolve_project_root() / args.config
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    run_sft_training(str(config_path))


if __name__ == "__main__":
    main()
