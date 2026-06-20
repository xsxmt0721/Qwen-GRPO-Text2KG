"""
test_node_reward.py
===================
测试 SFT 模型的 completion 并验证 NodeRewardCalculator 是否可跑通。

加载:
  - sft_node_train.json      → 全部标注样本 (含 train/val/test)
  - sft_node_split.json      → 划分索引 (train/val/test)
  - sft_node_test_raw.json   → SFT 模型对 test 子集的预测输出

通过 split 文件定位 test 子集在 train.json 中的正确行号，
确保每条预测 completion_hat 与其 ground-truth completion 正确配对。

运行方式（项目根目录）:
  python scripts/test_node_reward.py
  python scripts/test_node_reward.py -n 20          # 仅测试前 20 条
  python scripts/test_node_reward.py -n 20 --detail  # 打印每条详情
  python scripts/test_node_reward.py --control       # 对照测试: 用标注completion作为预测, 验证满分
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# ── 项目根路径 ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.nodes_reward import NodeRewardCalculator


# ================================================================
# 数据加载
# ================================================================

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载 JSONL 文件，每行一个 JSON 对象。"""
    data = []
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_json(file_path: str) -> Any:
    """加载 JSON 文件。"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ================================================================
# 主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test NodeRewardCalculator on SFT model outputs"
    )
    parser.add_argument(
        "-n", "--max-samples", type=int, default=None,
        help="Only test the first N samples (default: all).",
    )
    parser.add_argument(
        "--detail", action="store_true",
        help="Print per-sample detail.",
    )
    parser.add_argument(
        "--train-file", type=str,
        default="/workspace/Output/SFTDatasets/sft_node_train.json",
        help="Path to sft_node_train.json (all samples with ground-truth completions).",
    )
    parser.add_argument(
        "--split-file", type=str,
        default="/workspace/Output/SFTDatasets/sft_node_split.json",
        help="Path to sft_node_split.json (train/val/test index split).",
    )
    parser.add_argument(
        "--raw-file", type=str,
        default="/workspace/Output/SFTDatasets/sft_node_test_raw.json",
        help="Path to sft_node_test_raw.json (SFT model outputs on test set).",
    )
    parser.add_argument(
        "--config", type=str,
        default="config/reward.yaml",
        help="Path to reward.yaml config.",
    )
    parser.add_argument(
        "--control", action="store_true",
        help="Control test: use ground-truth completion as prediction "
             "(expect near-perfect scores). Raw file is ignored.",
    )
    args = parser.parse_args()

    # ── 加载数据 ──
    print(f"Loading all data: {args.train_file}")
    all_data = load_jsonl(args.train_file)
    print(f"  loaded {len(all_data)} samples")

    print(f"Loading split info: {args.split_file}")
    split_info = load_json(args.split_file)
    test_indices = split_info.get("test", [])
    print(f"  train={len(split_info.get('train', []))}, "
          f"val={len(split_info.get('val', []))}, "
          f"test={len(test_indices)}")

    if not test_indices:
        print("ERROR: No test indices found in split file.")
        sys.exit(1)

    if args.control:
        print("\n*** CONTROL MODE: using ground-truth completion as prediction ***\n")
        raw_data = None  # 不需要加载 raw_file
    else:
        print(f"Loading raw output data: {args.raw_file}")
        raw_data = load_jsonl(args.raw_file)
        print(f"  loaded {len(raw_data)} samples")

        if len(test_indices) != len(raw_data):
            print(f"WARNING: test indices ({len(test_indices)}) and raw outputs "
                  f"({len(raw_data)}) have different lengths! "
                  f"Using min({len(test_indices)}, {len(raw_data)}).")
            n = min(len(test_indices), len(raw_data))
            test_indices = test_indices[:n]
            raw_data = raw_data[:n]

    if args.max_samples is not None and args.max_samples > 0:
        test_indices = test_indices[:args.max_samples]
        if not args.control:
            raw_data = raw_data[:args.max_samples]
        print(f"  limiting to first {len(test_indices)} test samples")

    # ── 初始化奖励计算器 ──
    print(f"\nInitializing NodeRewardCalculator from: {args.config}")
    calc = NodeRewardCalculator(args.config)
    print()

    # ── 逐样本计算 ──
    results: List[Dict[str, Any]] = []
    detail_header = (
        f"{'#':>4s}  {'idx':>5s}  {'struct':>7s}  {'len_pen':>8s}  "
        f"{'text_match':>10s}  {'ds_match':>9s}  {'total':>8s}  "
        f"{'pred_cnt':>8s}  {'std_cnt':>7s}"
    )

    if args.detail:
        print(detail_header)
        print("-" * len(detail_header))

    for i, data_idx in enumerate(test_indices):
        if data_idx >= len(all_data):
            print(f"WARNING: index {data_idx} out of range ({len(all_data)}), skipping.")
            continue

        gt_sample = all_data[data_idx]
        text = gt_sample.get("text", "")
        completion_gt = gt_sample.get("completion", "")

        if args.control:
            # 对照测试: 用标注 completion 作为预测
            completion_hat = completion_gt
        else:
            raw_sample = raw_data[i]
            completion_hat = raw_sample.get("completion", "")

        result = calc.calculate_reward(completion_hat, completion_gt, text)
        results.append(result)

        if args.detail:
            print(
                f"{i:4d}  {data_idx:5d}  {result['structure_score']:7.4f}  "
                f"{result['length_penalty']:8.4f}  "
                f"{result['text_matching_score']:10.4f}  "
                f"{result['dataset_matching_score']:9.4f}  "
                f"{result['total_reward']:8.4f}  "
                f"{result['pred_count']:8d}  "
                f"{result['std_count']:7d}"
            )

    # ── 汇总统计 ──
    n = len(results)
    if n == 0:
        print("\nNo results to summarize.")
        return

    total_rewards = np.array([r["total_reward"] for r in results])
    struct_scores  = np.array([r["structure_score"] for r in results])
    len_penalties  = np.array([r["length_penalty"] for r in results])
    text_matches   = np.array([r["text_matching_score"] for r in results])
    ds_matches     = np.array([r["dataset_matching_score"] for r in results])
    pred_counts    = np.array([r["pred_count"] for r in results])
    std_counts     = np.array([r["std_count"] for r in results])

    zero_struct = int(np.sum(struct_scores == 0.0))

    print()
    print("=" * 65)
    print("  SUMMARY  (n = {})".format(n))
    print("=" * 65)
    print(f"  {'Metric':<25s} {'Mean':>9s}  {'Std':>9s}  {'Min':>9s}  {'Max':>9s}")
    print("-" * 65)

    rows = [
        ("total_reward",        total_rewards),
        ("structure_score",     struct_scores),
        ("length_penalty",      len_penalties),
        ("text_matching_score", text_matches),
        ("dataset_matching_score", ds_matches),
        ("pred_count",          pred_counts.astype(float)),
        ("std_count",           std_counts.astype(float)),
    ]

    for name, arr in rows:
        print(f"  {name:<25s} {np.mean(arr):9.4f}  {np.std(arr):9.4f}  "
              f"{np.min(arr):9.4f}  {np.max(arr):9.4f}")

    print("-" * 65)
    print(f"  Total samples:              {n}")
    print(f"  Zero-structure (invalid):   {zero_struct}  ({zero_struct/n*100:.1f}%)")
    print(f"  Non-zero structure (valid): {n - zero_struct}  ({(n-zero_struct)/n*100:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
