#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SFT 数据集划分脚本
------------------
从 config/data.yaml 读取划分比例与随机种子，
加载 Output/SFTDatasets/sft_node_train.json（JSONL 格式），
将样本索引按比例随机划分为 train / val / test，
最终仅将索引表保存到 Output/SFTDatasets/sft_node_split.json。

用法:
    python scripts/split_sft_dataset.py
    python scripts/split_sft_dataset.py --config config/data.yaml
    python scripts/split_sft_dataset.py -i Output/SFTDatasets/sft_node_train.json -o Output/SFTDatasets/sft_node_split.json
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
        description="将 SFT 数据集按配置比例划分为 train / val / test，仅保存索引"
    )
    parser.add_argument(
        "--config", "-c",
        default="config/data.yaml",
        help="YAML 配置文件路径 (默认: config/data.yaml)",
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="输入 JSONL 文件路径 (默认: Output/SFTDatasets/sft_node_train.json)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出 split JSON 文件路径 (默认: Output/SFTDatasets/sft_node_split.json)",
    )
    args = parser.parse_args()

    project_root = resolve_project_root()

    # ---- 读取配置 ----
    config_path = (project_root / args.config).resolve()
    if not config_path.is_file():
        print(f"[错误] 配置文件不存在: {config_path}")
        sys.exit(1)

    config = load_yaml_config(config_path)
    dataset_cfg: dict = config.get("dataset", {})

    train_ratio: float = float(dataset_cfg.get("train_split", 0.6))
    val_ratio: float = float(dataset_cfg.get("val_split", 0.1))
    test_ratio: float = float(dataset_cfg.get("test_split", 0.3))
    seed: int = int(dataset_cfg.get("seed", 42))

    print(f"配置读取成功: train={train_ratio}, val={val_ratio}, test={test_ratio}, seed={seed}")

    # ---- 输入文件 ----
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = project_root / "Output" / "SFTDatasets" / "sft_node_train.json"
    input_path = input_path.resolve()

    if not input_path.is_file():
        print(f"[错误] 输入文件不存在: {input_path}")
        sys.exit(1)

    print(f"数据集路径: {input_path}")

    # ---- 输出文件 ----
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / "Output" / "SFTDatasets" / "sft_node_split.json"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 统计行数 ----
    total = count_jsonl_lines(input_path)
    print(f"数据集总样本数: {total}")

    if total == 0:
        print("[错误] 数据集为空")
        sys.exit(1)

    # ---- 划分索引 ----
    split_result = split_indices(
        total=total,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    print(f"训练集 (train):  {len(split_result['train'])} 条")
    print(f"验证集 (val):    {len(split_result['val'])} 条")
    print(f"测试集 (test):   {len(split_result['test'])} 条")

    # ---- 保存 ----
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_result, f, ensure_ascii=False, indent=2)

    print(f"索引文件已保存: {output_path}")


if __name__ == "__main__":
    main()
