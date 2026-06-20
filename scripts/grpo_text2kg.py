"""
grpo_text2kg.py
===============
启动 Phase 1 节点提取模型 GRPO 强化微调训练。

运行方式（容器内）：
  python scripts/grpo_text2kg.py
  python scripts/grpo_text2kg.py --config config/grpo_nodes.yaml
  python scripts/grpo_text2kg.py -n 100              # 仅用前 100 条训练样本快速测试

前置条件：
  1. 已运行 scripts/build_sft_dataset.py  生成 grpo_node_train.json
  2. 已运行 scripts/split_sft_dataset.py -d grpo  生成 grpo_node_split.json
  3. SFT LoRA 权重 /models/qwen-sft-text2node 已就绪
  4. 基座模型 /models/Qwen2.5-1.5B-Instruct 已就绪

输出：
  /models/qwen-grpo-text2node/               — GRPO 微调后的 LoRA 权重
  /workspace/Logs/grpo_nodes/                — TensorBoard 训练日志
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.nodes_grpotrainer import run_grpo_training


def main():
    import argparse

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

    run_grpo_training(
        config_path=args.config,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )


if __name__ == "__main__":
    main()
