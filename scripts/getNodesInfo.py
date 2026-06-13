"""
getNodesInfo.py
---------------
从 Data/data/node.json 读取节点数据集，
统计类型/用途/疾病的节点数量，以及各维度覆盖情况。
"""

import json
import os

# ============================================================
# 可编辑配置
# ============================================================
NODE_FILE = os.path.join("Data", "data", "node.json")
INDEX_DIR = os.path.join("Data", "data", "index")
OUTPUT_DIR = os.path.join("scripts", "output")


def load_dataset() -> list:
    with open(NODE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def count_by_dimension(dataset: list, key: str) -> dict:
    """统计某个维度下各标签的节点数量，按数量降序排列。"""
    counts = {}
    for entry in dataset:
        for v in entry.get(key, []):
            counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def load_index_file(filepath: str) -> list:
    """加载单个索引 JSON 文件，返回 ID 列表。"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def inspect_index_files(dataset: list, index_dir: str):
    """
    遍历 index 目录下所有索引文件，打印每个文件的条目数，
    并与 node.json 中对应字段的条目数做对比。
    """
    if not os.path.isdir(index_dir):
        print(f"\n[警告] 索引目录不存在: {index_dir}")
        return

    index_files = sorted([
        f for f in os.listdir(index_dir) if f.endswith(".json")
    ])

    if not index_files:
        print(f"\n[警告] 索引目录为空: {index_dir}")
        return

    # 解析每个索引文件名: index_{dimension}_{value}.json
    import re
    file_pattern = re.compile(r"^index_(type|use|disease)_(.+)\.json$")

    print("\n" + "=" * 70)
    print("索引文件对照表".center(64))
    print("=" * 70)
    print(f"  {'索引文件':<40s} {'文件条目':>6s}  {'node.json':>8s}  {'差异':>6s}")
    print("-" * 70)

    issues = []  # 收集不一致项

    for filename in index_files:
        m = file_pattern.match(filename)
        if not m:
            continue

        dim = m.group(1)          # type / use / disease
        label = m.group(2)        # 标签名

        # 索引文件中的条目数
        filepath = os.path.join(index_dir, filename)
        idx_ids = load_index_file(filepath)
        file_count = len(idx_ids)

        # node.json 中该标签的条目数
        node_count = sum(
            1 for e in dataset
            if label in e.get(dim, [])
        )

        diff = file_count - node_count
        if diff == 0:
            flag = ""
        elif diff > 0:
            flag = f"  +{diff}"
        else:
            flag = f"  {diff}"

        print(f"  {filename:<40s} {file_count:>6d}  {node_count:>8d}  {flag:>6s}")

        if diff != 0:
            issues.append((filename, dim, label, file_count, node_count, diff))

    print("-" * 70)

    if issues:
        print(f"\n⚠ 不一致项 ({len(issues)} 个):")
        print(f"  {'文件':<42s} {'维度':>6s} {'索引':>6s} {'node.json':>8s} {'差异':>6s}")
        for filename, dim, label, fc, nc, d in issues:
            print(f"  {filename:<42s} {dim:>6s} {fc:>6d} {nc:>8d} {d:>+6d}")
    else:
        print(f"\n✅ 所有索引文件与 node.json 完全一致")

    print("=" * 70)


def count_coverage(dataset: list) -> dict:
    """
    统计各维度覆盖情况，返回结构化数据供保存。
    """
    total = len(dataset)

    all_three = sum(1 for e in dataset if e["type"] and e["use"] and e["disease"])
    only_type = sum(1 for e in dataset if e["type"] and not e["use"] and not e["disease"])
    only_use  = sum(1 for e in dataset if not e["type"] and e["use"] and not e["disease"])
    only_disease = sum(1 for e in dataset if not e["type"] and not e["use"] and e["disease"])
    type_and_use = sum(1 for e in dataset if e["type"] and e["use"] and not e["disease"])
    type_and_disease = sum(1 for e in dataset if e["type"] and not e["use"] and e["disease"])
    use_and_disease = sum(1 for e in dataset if not e["type"] and e["use"] and e["disease"])
    none_all = sum(1 for e in dataset if not e["type"] and not e["use"] and not e["disease"])

    result = {
        "total": total,
        "all_three": all_three,
        "none_all": none_all,
        "only_type": only_type,
        "only_use": only_use,
        "only_disease": only_disease,
        "type_and_use": type_and_use,
        "type_and_disease": type_and_disease,
        "use_and_disease": use_and_disease,
    }

    print(f"\n节点总数: {total}")
    print(f"  三项均不为空 (type+use+disease): {all_three}")
    print(f"  全部为空:                      {none_all}")
    print()
    print(f"  仅一项不为空:")
    print(f"    仅 type:    {only_type}")
    print(f"    仅 use:     {only_use}")
    print(f"    仅 disease: {only_disease}")
    print(f"    (小计)      {only_type + only_use + only_disease}")
    print()
    print(f"  某两项不为空:")
    print(f"    type + use:       {type_and_use}")
    print(f"    type + disease:   {type_and_disease}")
    print(f"    use  + disease:   {use_and_disease}")
    print(f"    (小计)            {type_and_use + type_and_disease + use_and_disease}")

    return result


def collect_no_type_nodes(dataset: list) -> list:
    """
    收集疾病/用途非空但类型为空的节点。
    即: (use 或 disease 非空) 且 type 为空。
    """
    return [
        e for e in dataset
        if not e["type"] and (e["use"] or e["disease"])
    ]


def collect_all_three_nodes(dataset: list) -> list:
    """收集三项标签均不为空的节点。"""
    return [
        e for e in dataset
        if e["type"] and e["use"] and e["disease"]
    ]


def save_outputs(dataset: list, type_counts: dict, use_counts: dict,
                 disease_counts: dict, coverage: dict,
                 output_dir: str):
    """将所有结果保存到 scripts/output 目录。"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 完整结构化报告 (JSON)
    report = {
        "total_nodes": len(dataset),
        "by_type": type_counts,
        "by_use": use_counts,
        "by_disease": disease_counts,
        "coverage": coverage,
    }
    report_path = os.path.join(output_dir, "nodes_info_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n完整报告: {report_path}")

    # 2. 疾病/用途非空但类型为空的节点
    no_type_nodes = collect_no_type_nodes(dataset)
    path1 = os.path.join(output_dir, "nodes_no_type.json")
    with open(path1, "w", encoding="utf-8") as f:
        json.dump(no_type_nodes, f, ensure_ascii=False, indent=2)
    print(f"类型为空的节点 (use/disease 非空): {path1} ({len(no_type_nodes)} 个)")

    # 3. 三项均非空的节点
    three_dim_nodes = collect_all_three_nodes(dataset)
    path2 = os.path.join(output_dir, "nodes_all_three_dim.json")
    with open(path2, "w", encoding="utf-8") as f:
        json.dump(three_dim_nodes, f, ensure_ascii=False, indent=2)
    print(f"三项均非空节点: {path2} ({len(three_dim_nodes)} 个)")

    # 4. 人类可读文本报告（终端打印内容的快照）
    txt_path = os.path.join(output_dir, "nodes_info_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"节点数据集信息报告\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"节点总数: {len(dataset)}\n\n")

        f.write(f"按 type（类型）统计:\n")
        for name, cnt in type_counts.items():
            f.write(f"  {name}: {cnt}\n")

        f.write(f"\n按 use（用途）统计:\n")
        for name, cnt in use_counts.items():
            f.write(f"  {name}: {cnt}\n")

        f.write(f"\n按 disease（疾病）统计:\n")
        for name, cnt in disease_counts.items():
            f.write(f"  {name}: {cnt}\n")

        f.write(f"\n维度覆盖情况:\n")
        f.write(f"  总节点数: {coverage['total']}\n")
        f.write(f"  三项均不为空: {coverage['all_three']}\n")
        f.write(f"  全部为空:     {coverage['none_all']}\n")
        f.write(f"  仅 type:      {coverage['only_type']}\n")
        f.write(f"  仅 use:       {coverage['only_use']}\n")
        f.write(f"  仅 disease:   {coverage['only_disease']}\n")
        f.write(f"  type+use:     {coverage['type_and_use']}\n")
        f.write(f"  type+disease: {coverage['type_and_disease']}\n")
        f.write(f"  use+disease:  {coverage['use_and_disease']}\n")

        f.write(f"\n类型为空的节点 (共 {len(no_type_nodes)} 个):\n")
        for e in no_type_nodes:
            f.write(f"  {e['name']:60s} use={e['use']} disease={e['disease']}\n")

        f.write(f"\n三项均非空节点 (共 {len(three_dim_nodes)} 个):\n")
        for e in three_dim_nodes:
            f.write(f"  {e['name']:60s} type={e['type']} use={e['use']} disease={e['disease']}\n")

    print(f"文本报告: {txt_path}")


def print_three_dim_nodes(dataset: list):
    """打印三项标签均不为空的节点及其详细信息。"""
    matched = [e for e in dataset if e["type"] and e["use"] and e["disease"]]
    print(f"\n三项均不为空的节点 (共 {len(matched)} 个):")
    print("-" * 70)
    for i, e in enumerate(matched, 1):
        print(f"  [{i}] name={e['name']}")
        print(f"       type=    {e['type']}")
        print(f"       use=     {e['use']}")
        print(f"       disease= {e['disease']}")
        print()


def main():
    print(f"读取数据集: {NODE_FILE}")
    dataset = load_dataset()
    print(f"节点总数: {len(dataset)}")

    # 按类型统计
    print("\n" + "=" * 50)
    print("按 type（类型）统计:")
    print("=" * 50)
    type_counts = count_by_dimension(dataset, "type")
    for name, cnt in type_counts.items():
        print(f"  {name}: {cnt}")

    # 按用途统计
    print("\n" + "=" * 50)
    print("按 use（用途）统计:")
    print("=" * 50)
    use_counts = count_by_dimension(dataset, "use")
    for name, cnt in use_counts.items():
        print(f"  {name}: {cnt}")

    # 按疾病统计
    print("\n" + "=" * 50)
    print("按 disease（疾病）统计:")
    print("=" * 50)
    disease_counts = count_by_dimension(dataset, "disease")
    for name, cnt in disease_counts.items():
        print(f"  {name}: {cnt}")

    # 索引文件对照
    inspect_index_files(dataset, INDEX_DIR)

    # 覆盖情况
    print("\n" + "=" * 50)
    print("维度覆盖情况:")
    print("=" * 50)
    coverage = count_coverage(dataset)

    # 保存输出
    save_outputs(dataset, type_counts, use_counts, disease_counts, coverage, OUTPUT_DIR)


if __name__ == "__main__":
    main()
