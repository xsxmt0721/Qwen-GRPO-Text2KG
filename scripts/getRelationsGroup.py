"""
getRelationsGroup.py
--------------------
将 Data/graph/{disease}/ 目录下的关系数据整合为统一的关系数据集。

关系文件格式 (JSON 字典):
  "<snode_1>+<snode_2>+...": [<tnode_1>, <tnode_2>, ...]

输出：
  /data/data/relation.json
    每条关系的结构:
    {
      "source_node": [<source_node1>, ...],   # 按 + 拆分
      "source_id":   [<source_id1>, ...],     # 在 node.json 中的索引, -1 表示未找到
      "target_node": [<target_node1>, ...],   # 按 + 拆分
      "target_id":   [<target_id1>, ...],
      "disease": <disease>,
      "use": <use>
    }
  注意：源数据中每个 tnode 对应一个独立的关系（共享同一 source）。
"""

import json
import os
import re
from collections import defaultdict

# ============================================================
# 可编辑配置
# ============================================================
GRAPH_DIR = os.path.join("Data", "graph")         # 原始数据目录
NODE_FILE = os.path.join("Data", "data", "node.json")
OUTPUT_RELATION_FILE = os.path.join("Data", "data", "relation.json")
OUTPUT_INDEX_DIR = os.path.join("Data", "data", "index")

# ============================================================
# 工具函数
# ============================================================

def extract_use_from_rel_filename(filename: str) -> str:
    """
    从关系文件名中提取 use 标签。
    格式: KG-{use}xxx-关系数量{count}.txt
    例如: KG-临床诊断140-关系数量154.txt → 临床诊断
          KG-治疗253-关系数量941975.txt  → 治疗
    """
    m = re.match(r"KG-(.+?)\d+-关系数量\d+\.txt$", filename)
    if m:
        return m.group(1)
    return None


def is_relation_file(filename: str) -> bool:
    """判断是否为关系文件"""
    return "关系数量" in filename and filename.endswith(".txt")


def load_node_dataset() -> list:
    """加载 node.json"""
    with open(NODE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_name_to_id_map(dataset: list) -> dict:
    """构建 node_name → index 的查找字典"""
    return {entry["name"]: idx for idx, entry in enumerate(dataset)}


def split_by_plus(text: str) -> list:
    """按 + 切分并去除空白和空串"""
    return [s.strip() for s in text.split("+") if s.strip()]


def lookup_ids(names: list, name_to_id: dict) -> list:
    """将节点名称列表转换为 id 列表，未找到的返回 -1"""
    return [name_to_id.get(n, -1) for n in names]


def process_relations(graph_dir: str, node_dataset: list) -> list:
    """
    遍历 disease 子目录下的关系文件，构建关系数据集。

    返回: list[dict]
    """
    name_to_id = build_name_to_id_map(node_dataset)
    missing_nodes = []  # [(role, node_name, filename, disease, use), ...]
    all_relations = []

    # 获取所有 disease 子目录
    disease_dirs = []
    for item in os.listdir(graph_dir):
        item_path = os.path.join(graph_dir, item)
        if os.path.isdir(item_path):
            disease_dirs.append(item)
    print(f"发现疾病子目录: {disease_dirs}")

    total_files = 0
    total_raw_edges = 0  # JSON 中的原始 key 数量
    total_relations = 0  # 展开后的关系数

    for disease_name in sorted(disease_dirs):
        disease_path = os.path.join(graph_dir, disease_name)
        print(f"\n--- 处理疾病子目录: {disease_name} ---")

        for filename in sorted(os.listdir(disease_path)):
            filepath = os.path.join(disease_path, filename)
            if not os.path.isfile(filepath):
                continue
            if not is_relation_file(filename):
                continue

            use_label = extract_use_from_rel_filename(filename)
            if use_label is None:
                print(f"  [跳过] 无法解析用途: {filename}")
                continue

            total_files += 1

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            edge_count = len(data)
            rel_count = 0
            total_raw_edges += edge_count

            for source_key, target_list in data.items():
                # 按 + 拆分 source
                source_nodes = split_by_plus(source_key)
                source_ids = lookup_ids(source_nodes, name_to_id)

                # 记录缺失的 source 节点
                for sn, sid in zip(source_nodes, source_ids):
                    if sid == -1:
                        missing_nodes.append(
                            ("source", sn, filename, disease_name, use_label)
                        )

                # 每个 tnode 创建一个独立的关系
                for tnode in target_list:
                    target_nodes = split_by_plus(tnode)
                    target_ids = lookup_ids(target_nodes, name_to_id)

                    # 记录缺失的 target 节点
                    for tn, tid in zip(target_nodes, target_ids):
                        if tid == -1:
                            missing_nodes.append(
                                ("target", tn, filename, disease_name, use_label)
                            )

                    all_relations.append({
                        "source_node": source_nodes,
                        "source_id": source_ids,
                        "target_node": target_nodes,
                        "target_id": target_ids,
                        "disease": disease_name,
                        "use": use_label,
                    })
                    rel_count += 1

            total_relations += rel_count
            print(f"  {filename}: use={use_label}, 原始边={edge_count}, 展开后关系={rel_count}")

    # 打印缺失节点报告
    if missing_nodes:
        print(f"\n{'='*60}")
        print(f"⚠ 无法查询到 id 的节点 (共 {len(missing_nodes)} 处):")
        print(f"{'='*60}")
        # 去重并按名称排序
        unique_missing = defaultdict(list)
        for role, name, fname, disease, use in missing_nodes:
            unique_missing[(role, name)].append((fname, disease, use))
        for (role, name), occurrences in sorted(unique_missing.items()):
            print(f"  [{role}] '{name}' — 出现 {len(occurrences)} 次")
            for fname, disease, use in occurrences[:3]:  # 最多显示 3 处
                print(f"        文件: {fname}, disease={disease}, use={use}")
            if len(occurrences) > 3:
                print(f"        ... 等 {len(occurrences)} 处")
    else:
        print("\n✓ 所有节点均可查询到 id")

    print(f"\n总计: {total_files} 个关系文件, {total_raw_edges} 条原始边, {total_relations} 条展开关系")
    return all_relations


def build_index(dataset: list, key: str) -> dict:
    """
    为数据集按指定 key 建立索引。
    key 为单值字段（如 "disease", "use"），非列表字段。
    """
    index = defaultdict(list)
    for idx, entry in enumerate(dataset):
        v = entry.get(key)
        if v:
            index[v].append(idx)
    return dict(index)


def save_index_files(index: dict, key: str, output_dir: str):
    """将单个维度的索引保存为多个 JSON 文件。"""
    for value, indices in index.items():
        safe_value = value.replace("/", "_").replace("\\", "_").replace(":", "_")
        filename = f"rel_index_{key}_{safe_value}.json"    # rel_ 前缀避免与 node 索引冲突
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(indices, f, ensure_ascii=False)


# ============================================================
# 主流程
# ============================================================

def main():
    print(f"节点数据集: {NODE_FILE}")
    print(f"关系数据目录: {GRAPH_DIR}")
    print(f"输出文件: {OUTPUT_RELATION_FILE}")

    # 加载节点数据
    node_dataset = load_node_dataset()
    print(f"已加载 {len(node_dataset)} 个节点")

    # 处理关系
    print("\n" + "=" * 60)
    print("开始构建关系数据集...")
    print("=" * 60)
    relations = process_relations(GRAPH_DIR, node_dataset)

    # 保存关系数据集
    os.makedirs(os.path.dirname(OUTPUT_RELATION_FILE), exist_ok=True)
    with open(OUTPUT_RELATION_FILE, "w", encoding="utf-8") as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)
    print(f"\n关系数据集已保存: {OUTPUT_RELATION_FILE}")
    print(f"共 {len(relations)} 条关系")

    # 建立并保存索引 (按 disease, use)
    os.makedirs(OUTPUT_INDEX_DIR, exist_ok=True)
    for key in ["disease", "use"]:
        index = build_index(relations, key)
        save_index_files(index, key, OUTPUT_INDEX_DIR)
        print(f"关系索引 (key={key}) 已保存至: {OUTPUT_INDEX_DIR}/ (共 {len(index)} 个文件)")

    print("\n完成!")


if __name__ == "__main__":
    main()
