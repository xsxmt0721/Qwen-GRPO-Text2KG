"""
build_sft_dataset.py
====================
构建 Phase 0 SFT 节点提取训练数据集。

运行方式（容器内）：
  python scripts/build_sft_dataset.py

输出：
  /workspace/Output/SFTDatasets/sft_node_train.json  — SFT 训练数据 (JSONL)
  scripts/output/sft_dataset_report.json             — 构建报告
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sft_datasets import build_sft_dataset


def main():
    build_sft_dataset(config_path="config/data.yaml")


if __name__ == "__main__":
    main()
