"""
sft_text2kg_eval.py
===================
启动 Phase 0 节点提取模型 SFT 测试评估。

运行方式（容器内）：
  python scripts/sft_text2kg_eval.py
  python scripts/sft_text2kg_eval.py --config config/sft.yaml

前置条件：
  1. 已运行 scripts/sft_text2kg.py  完成 SFT 微调
  2. LoRA 权重已保存在 /models/qwen-sft-text2node/
  3. 测试集划分已在 sft_node_split.json 中

输出：
  /workspace/Output/SFTDatasets/sft_node_test_details.json  — 测试集逐样本详情
  /workspace/Output/SFTDatasets/sft_node_test_output.json   — 测试集聚合指标
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.eval_node_extract_sft import run_evaluation


def main():
    run_evaluation(config_path="config/sft.yaml")


if __name__ == "__main__":
    main()
