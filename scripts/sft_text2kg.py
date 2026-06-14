"""
sft_text2kg.py
==============
启动 Phase 0 节点提取模型 SFT 监督微调训练。

运行方式（容器内）：
  python scripts/sft_text2kg.py
  python scripts/sft_text2kg.py --config config/sft.yaml

前置条件：
  1. 已运行 scripts/build_sft_dataset.py  生成 sft_node_train.json
  2. 已运行 scripts/split_sft_dataset.py  生成 sft_node_split.json
  3. 基座模型 /models/Qwen2.5-1.5B-Instruct 已就绪

输出：
  /models/qwen-sft-text2node/                — 微调后的 LoRA 权重
  /workspace/Logs/sft_text2node/             — TensorBoard 训练日志
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.nodes_extract_sft_trainer import run_sft_training


def main():
    run_sft_training(config_path="config/sft.yaml")


if __name__ == "__main__":
    main()
