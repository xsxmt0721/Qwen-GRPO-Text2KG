#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集划分脚本 (SFT / GRPO)
----------------------------
从 config/data.yaml 读取指定数据集的划分比例与随机种子，
加载对应目录下的 *_node_train.json（JSONL 格式），
将样本索引按比例随机划分为 train / val / test，
最终将索引表保存到同一目录下的 *_node_split.json。

用法:
    python scripts/split_sft_dataset.py                        # 默认划分 SFT
    python scripts/split_sft_dataset.py --dataset sft          # 划分 SFT
    python scripts/split_sft_dataset.py -d grpo                # 划分 GRPO
    python scripts/split_sft_dataset.py -d grpo -c config/data.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


def resolve_project_root() -> Path:
    """返回项目根目录（scripts/ 的父目录）。"""
    return Path(__file__).resolve().parent.parent


def load_yaml_config(config_path: Path) -> dict:
    """读取 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_jsonl_lines(file_path: Path) -> int:
    """统计 JSONL 文件的总行数（每行一个 JSON 对象）。"""
    n = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def split_indices(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[int]]:
    """
    将 0 … total-1 的整数索引随机打乱后按比例切分为 train / val / test。

    Returns
    -------
    dict
        {"train": [...], "val": [...], "test": [...]}
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"划分比例之和应为 1.0，当前为 {ratio_sum}")

    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": indices[:train_end].tolist(),
        "val": indices[train_end:val_end].tolist(),
        "test": indices[val_end:].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 SFT / GRPO 数据集按配置比例划分为 train / val / test，仅保存索引"
    )
    parser.add_argument(
        "--dataset", "-d",
        default="sft",
        choices=["sft", "grpo"],
        help="选择要划分的数据集: sft 或 grpo (默认: sft)",
    )
    parser.add_argument(
        "--config", "-c",
        default="config/data.yaml",
        help="YAML 配置文件路径 (默认: config/data.yaml)",
    )
    args = parser.parse_args()

    project_root = resolve_project_root()

    # ── 读取配置 ──
    config_path = (project_root / args.config).resolve()
    if not config_path.is_file():
        print(f"[错误] 配置文件不存在: {config_path}")
        sys.exit(1)

    config = load_yaml_config(config_path)

    # 根据 --dataset 选择配置段
    if args.dataset == "sft":
        cfg_key = "dataset"
        prefix = "sft"
    else:
        cfg_key = "dataset_grpo"
        prefix = "grpo"

    dataset_cfg: dict = config.get(cfg_key, {})
    train_ratio: float = float(dataset_cfg.get("train_split", 0.6))
    val_ratio: float = float(dataset_cfg.get("val_split", 0.1))
    test_ratio: float = float(dataset_cfg.get("test_split", 0.3))
    seed: int = int(dataset_cfg.get("seed", 42))
    output_dir: str = str(dataset_cfg.get("output_dir", f"/workspace/Output/{prefix.upper()}Datasets"))

    print(f"[{prefix.upper()}] 配置: train={train_ratio}, val={val_ratio}, test={test_ratio}, seed={seed}")

    # ── 输入文件 ──
    input_path = (Path(output_dir) / f"{prefix}_node_train.json").resolve()
    if not input_path.is_file():
        print(f"[错误] 输入文件不存在: {input_path}")
        sys.exit(1)

    print(f"[{prefix.upper()}] 数据集路径: {input_path}")

    # ── 输出文件 ──
    output_path = (Path(output_dir) / f"{prefix}_node_split.json").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 统计行数 ──
    total = count_jsonl_lines(input_path)
    print(f"[{prefix.upper()}] 数据集总样本数: {total}")

    if total == 0:
        print(f"[错误] 数据集为空: {input_path}")
        sys.exit(1)

    # ── 划分索引 ──
    split_result = split_indices(
        total=total,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    print(f"  训练集 (train):  {len(split_result['train'])} 条")
    print(f"  验证集 (val):    {len(split_result['val'])} 条")
    print(f"  测试集 (test):   {len(split_result['test'])} 条")

    # ── 保存 ──
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_result, f, ensure_ascii=False, indent=2)

    print(f"[{prefix.upper()}] 索引文件已保存: {output_path}")


if __name__ == "__main__":
    main()
