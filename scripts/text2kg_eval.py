"""
sft_text2kg_eval.py
===================
启动 Phase 0 节点提取模型 SFT 测试评估。

运行方式（容器内）：
  python scripts/sft_text2kg_eval.py
  python scripts/sft_text2kg_eval.py --config config/sft.yaml
  python scripts/sft_text2kg_eval.py -n 50           # 仅评估前 50 个测试样本

前置条件：
  1. 已运行 scripts/sft_text2kg.py  完成 SFT 微调
  2. LoRA 权重已保存在 /models/qwen-sft-text2node/
  3. 测试集划分已在 sft_node_split.json 中

输出：
  /workspace/Output/SFTDatasets/sft_node_test_raw.json    — 测试集原始输出 (逐样本)
  /workspace/Output/SFTDatasets/sft_node_test_details.json  — 测试集逐样本详情
  /workspace/Output/SFTDatasets/sft_node_test_output.json   — 测试集聚合指标
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.eval_node_extract_sft import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="SFT Node Extraction Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="config/sft.yaml",
        help="Path to sft.yaml config file (default: config/sft.yaml)",
    )
    parser.add_argument(
        "-n",
        "--max-samples",
        type=int,
        default=None,
        help="Only evaluate the first N test samples (for quick testing).",
    )
    parser.add_argument(
        "--disable-sampling",
        action="store_true",
        help="Use greedy decoding instead of sampling.",
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

    import os
    if args.disable_sampling:
        os.environ["EVAL_DISABLE_SAMPLING"] = "1"

    run_evaluation(
        config_path=args.config,
        max_samples=args.max_samples,
        checkpoint_step=args.checkpoint,
    )


if __name__ == "__main__":
    main()
