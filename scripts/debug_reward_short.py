"""
debug_reward_short.py
=====================
调试脚本：模拟各种极短输出，验证在不同 ground truth 下的奖励分数。
用于排查 GRPO 训练中 reward=0.90 / length=2.0 的根因。

用法:
    python scripts/debug_reward_short.py
    python scripts/debug_reward_short.py --dataset Output/GRPODatasets/grpo_node_train.json -n 50
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.nodes_reward import NodeRewardCalculator


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Debug: test various short outputs against GRPO reward function"
    )
    parser.add_argument(
        "--dataset", type=str,
        default="Output/GRPODatasets/grpo_node_train.json",
    )
    parser.add_argument("-n", "--max-samples", type=int, default=50)
    parser.add_argument("--config", type=str, default="config/reward.yaml")
    args = parser.parse_args()

    # 加载真实数据
    data = load_jsonl(args.dataset)[:args.max_samples]
    print(f"Loaded {len(data)} samples from {args.dataset}")

    # 统计空标签占比
    empty_gt = sum(1 for d in data if '"nodes": []' in d.get("completion", "") or '"nodes":[]' in d.get("completion", ""))
    print(f"Empty GT samples: {empty_gt}/{len(data)} ({empty_gt/len(data)*100:.1f}%)")
    print()

    calc = NodeRewardCalculator(args.config)

    # 各种短输出候选项
    short_outputs = [
        ("[]",                        "裸空数组"),
        ("{}",                        "裸空对象"),
        ('{"nodes":[]}',              "标准空 JSON"),
        ('{"nodes": []}',             "标准空 JSON (带空格)"),
        ('```json\n{"nodes":[]}\n```', "带 markdown 包裹"),
        ('```json\n[]\n```',          "markdown + 裸数组"),
        ('\n',                        "仅换行"),
        ('',                          "空字符串"),
    ]

    # 按 ground truth 是否为空分组统计
    results_empty: Dict[str, List[float]] = {label: [] for _, label in short_outputs}
    results_nonempty: Dict[str, List[float]] = {label: [] for _, label in short_outputs}

    for sample in data:
        completion_gt = sample.get("completion", "")
        text = sample.get("text", "")
        is_empty_gt = '"nodes": []' in completion_gt or '"nodes":[]' in completion_gt

        for short_out, label in short_outputs:
            result = calc.calculate_reward(short_out, completion_gt, text)
            total = result["total_reward"]
            if is_empty_gt:
                results_empty[label].append(total)
            else:
                results_nonempty[label].append(total)

    # ── 打印结果 ──
    header = f"  {'Output':<35s} {'Empty GT':>10s} {'NonEmpty GT':>12s} {'Weighted Avg':>13s}"
    print(header)
    print("-" * len(header))

    empty_ratio = empty_gt / len(data)

    for short_out, label in short_outputs:
        avg_empty = np.mean(results_empty[label]) if results_empty[label] else float('nan')
        avg_nonempty = np.mean(results_nonempty[label]) if results_nonempty[label] else float('nan')
        weighted = empty_ratio * avg_empty + (1 - empty_ratio) * avg_nonempty

        print(f"  {label:<35s} {avg_empty:10.4f} {avg_nonempty:12.4f} {weighted:13.4f}")

    print()
    print(f"  Weighted avg uses empty_ratio = {empty_ratio:.3f}")

    # ── 单独分析 struct / text / ds 子分数 ──
    print()
    print("=" * 70)
    print("  Sub-score breakdown (using first non-empty GT sample)")
    print("=" * 70)

    # 找一个有节点的样本
    nonempty_sample = None
    for s in data:
        if '"nodes": []' not in s.get("completion", "") and '"nodes":[]' not in s.get("completion", ""):
            nonempty_sample = s
            break

    if nonempty_sample:
        print(f"  GT preview: {nonempty_sample['completion'][:100]}...")
        print()
        print(f"  {'Output':<35s} {'struct':>8s} {'len_pen':>8s} {'text':>8s} {'ds':>8s} {'total':>8s}")
        print("-" * 80)
        for short_out, label in short_outputs:
            result = calc.calculate_reward(
                short_out, nonempty_sample["completion"], nonempty_sample.get("text", "")
            )
            print(f"  {label:<35s} "
                  f"{result['structure_score']:8.4f} "
                  f"{result['length_penalty']:8.4f} "
                  f"{result['text_matching_score']:8.4f} "
                  f"{result['dataset_matching_score']:8.4f} "
                  f"{result['total_reward']:8.4f}")

    # ── 同样对空 GT ──
    empty_sample = None
    for s in data:
        if '"nodes": []' in s.get("completion", "") or '"nodes":[]' in s.get("completion", ""):
            empty_sample = s
            break

    if empty_sample:
        print()
        print(f"  GT preview (empty): {empty_sample['completion'][:80]}...")
        print()
        print(f"  {'Output':<35s} {'struct':>8s} {'len_pen':>8s} {'text':>8s} {'ds':>8s} {'total':>8s}")
        print("-" * 80)
        for short_out, label in short_outputs:
            result = calc.calculate_reward(
                short_out, empty_sample["completion"], empty_sample.get("text", "")
            )
            print(f"  {label:<35s} "
                  f"{result['structure_score']:8.4f} "
                  f"{result['length_penalty']:8.4f} "
                  f"{result['text_matching_score']:8.4f} "
                  f"{result['dataset_matching_score']:8.4f} "
                  f"{result['total_reward']:8.4f}")


if __name__ == "__main__":
    main()
