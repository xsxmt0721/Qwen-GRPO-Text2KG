"""
completion_length_stats.py
==========================
统计数据集中 completion 字段 tokenize 后的长度分布。

用法 (项目根目录):
    python scripts/completion_length_stats.py Output/GRPODatasets/grpo_node_train.json
    python scripts/completion_length_stats.py Output/GRPODatasets/grpo_node_train.json --bins 20
    python scripts/completion_length_stats.py Output/GRPODatasets/grpo_node_train.json --model /models/Qwen2.5-1.5B-Instruct
    python scripts/completion_length_stats.py Output/GRPODatasets/grpo_node_train.json --detail  # 打印每条
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


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载 JSONL 文件。"""
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


def load_tokenizer(model_path: str):
    """延迟导入，仅在需要时加载 tokenizer。"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="统计数据集中 completion 字段 tokenize 后的长度分布"
    )
    parser.add_argument(
        "dataset", type=str,
        help="数据集路径 (JSONL 格式)",
    )
    parser.add_argument(
        "--model", type=str,
        default="/models/Qwen2.5-1.5B-Instruct",
        help="Tokenizer 模型路径 (默认: /models/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--bins", type=int, default=15,
        help="直方图分箱数 (默认: 15)",
    )
    parser.add_argument(
        "--detail", action="store_true",
        help="打印每条样本的详细 token 长度。",
    )
    parser.add_argument(
        "--no-tokenize", action="store_true",
        help="不做 tokenize，仅统计字符长度 (快速模式)。",
    )
    args = parser.parse_args()

    # ── 加载数据 ──
    print(f"Loading dataset: {args.dataset}")
    data = load_jsonl(args.dataset)
    print(f"  Total samples: {len(data)}")

    # ── 加载 tokenizer ──
    if not args.no_tokenize:
        print(f"Loading tokenizer from: {args.model}")
        tokenizer = load_tokenizer(args.model)
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print()
    else:
        tokenizer = None
        print("  (fast mode: character length only)")
        print()

    # ── 统计 ──
    lengths = []
    empty_count = 0

    if args.detail:
        header = f"{'#':>5s}  {'length':>7s}  {'node_count':>10s}  completion_preview"
        print(header)
        print("-" * len(header))

    for i, sample in enumerate(data):
        completion = sample.get("completion", "")

        if tokenizer is not None:
            tokens = tokenizer.encode(completion, add_special_tokens=False)
            length = len(tokens)
        else:
            length = len(completion)

        lengths.append(length)

        if length == 0:
            empty_count += 1

        if args.detail:
            # 统计该 completion 中的节点数
            try:
                # 提取 JSON 并解析
                import re as _re
                s = completion.strip()
                m = _re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', s, _re.DOTALL)
                json_str = (m.group(1) if m else s).strip()
                obj = json.loads(json_str)
                if isinstance(obj, dict) and "nodes" in obj:
                    node_count = len(obj["nodes"])
                elif isinstance(obj, list):
                    node_count = len(obj)
                else:
                    node_count = -1
            except Exception:
                node_count = -1

            preview = completion.replace("\n", "\\n")[:80]
            print(f"{i:5d}  {length:7d}  {node_count:10d}  {preview}")

    lengths = np.array(lengths)

    # ── 汇总统计 ──
    print()
    print("=" * 60)
    print("Token Length Distribution" if tokenizer else "Character Length Distribution")
    print("=" * 60)
    print(f"  Total samples:    {len(lengths):>8d}")
    print(f"  Empty completions:{empty_count:>8d}  ({empty_count / len(lengths) * 100:.1f}%)")
    print(f"  Mean:             {np.mean(lengths):>8.1f}")
    print(f"  Std:              {np.std(lengths):>8.1f}")
    print(f"  Min:              {np.min(lengths):>8d}")
    print(f"  25% (Q1):         {np.percentile(lengths, 25):>8.0f}")
    print(f"  50% (Median):     {np.percentile(lengths, 50):>8.0f}")
    print(f"  75% (Q3):         {np.percentile(lengths, 75):>8.0f}")
    print(f"  Max:              {np.max(lengths):>8d}")
    print()

    # ── 分箱直方图 ──
    bins = args.bins
    counts, bin_edges = np.histogram(lengths, bins=bins)
    print(f"{'Range':>16s}  {'Count':>8s}  {'Ratio':>8s}  Histogram")
    print("-" * 60)

    max_count = max(counts) if max(counts) > 0 else 1
    bar_width = 30

    for i in range(len(counts)):
        lo = int(bin_edges[i])
        hi = int(bin_edges[i + 1])
        cnt = counts[i]
        ratio = cnt / len(lengths) * 100
        bar = "█" * int(cnt / max_count * bar_width)
        print(f"  [{lo:>4d}, {hi:>4d}]  {cnt:>8d}  {ratio:>7.1f}%  {bar}")

    print()

    # ── 按节点数分组的 token 长度分布 ──
    print("=" * 60)
    print("Token Length by Node Count")
    print("=" * 60)

    node_lengths: Dict[int, List[int]] = {}  # node_count → list of token lengths
    import re as _re

    for sample in data:
        completion = sample.get("completion", "")
        try:
            s = completion.strip()
            m = _re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', s, _re.DOTALL)
            json_str = (m.group(1) if m else s).strip()
            obj = json.loads(json_str)
            if isinstance(obj, dict) and "nodes" in obj:
                nc = len(obj["nodes"])
            elif isinstance(obj, list):
                nc = len(obj)
            else:
                nc = -1
        except Exception:
            nc = -1

        if tokenizer is not None:
            tok_len = len(tokenizer.encode(completion, add_special_tokens=False))
        else:
            tok_len = len(completion)

        node_lengths.setdefault(nc, []).append(tok_len)

    print(f"  {'Nodes':>6s}  {'Samples':>8s}  {'Mean Len':>10s}  {'Min':>6s}  {'Max':>6s}")
    print("-" * 50)
    for nc in sorted(node_lengths.keys()):
        arr = np.array(node_lengths[nc])
        label = str(nc) if nc >= 0 else "parse_err"
        print(f"  {label:>6s}  {len(arr):>8d}  {np.mean(arr):>10.1f}  {np.min(arr):>6.0f}  {np.max(arr):>6.0f}")


if __name__ == "__main__":
    main()
