"""
getNodesGroup.py
---------------
将 Data/graph 目录下散落的节点数据整理为统一的数据集。

目录结构说明：
  Data/graph/
    KG-{type}-节点数量{count}.txt        → 按 type 分类的节点
    高血压/
      KG-{use}-节点数量{count}.txt       → 高血压下按 use 分类的节点
    糖尿病/
      KG-{use}-节点数量{count}.txt       → 糖尿病下按 use 分类的节点

每个 txt 文件是一个 JSON 列表，末尾的 "dic num:XX" 不是节点，需过滤。

输出：
  Data/data/node.json                    → 主数据集
  Data/data/index/index_type_{type}.json → 按 type 的索引
  Data/data/index/index_use_{use}.json   → 按 use 的索引
  Data/data/index/index_disease_{disease}.json → 按 disease 的索引
"""

import json
import os
import re
from collections import defaultdict

# ============================================================
# 可编辑配置
# ============================================================
GRAPH_DIR = os.path.join("Data", "graph")       # 原始数据目录
OUTPUT_NODE_FILE = os.path.join("Data", "data", "node.json")
OUTPUT_INDEX_DIR = os.path.join("Data", "data", "index")

# Jaccard 补丁阈值：Jaccard >= 此值时才复用 type
JACCARD_PATCH_THRESHOLD = 0.6

# 补丁模式: "symmetric"(对称) 或 "asymmetric"(非对称)
#   对称:   Jaccard = |A ∩ B| / |A ∪ B|  — 双向相似，更保守
#   非对称: Jaccard = |A ∩ B| / |A|      — 仅要求 A 的 token 在 B 中覆盖率高，更激进
JACCARD_PATCH_MODE = "asymmetric"

# ============================================================
# 工具函数
# ============================================================

def extract_type_from_filename(filename: str) -> str:
    """
    从文件名中提取 type/use 标签。
    格式: KG-{标签}-节点数量{数字}.txt
    例如: KG-个人史概念-节点数量21.txt → 个人史概念
          KG-临床诊断-节点数量82.txt   → 临床诊断
    """
    m = re.match(r"KG-(.+)-节点数量\d+\.txt$", filename)
    if m:
        return m.group(1)
    return None


def is_node_file(filename: str) -> bool:
    """判断是否为节点文件（排除关系文件）"""
    return "关系数量" not in filename and filename.endswith(".txt")


def parse_node_file(filepath: str) -> list:
    """
    解析节点文件，返回节点名称列表（过滤掉末尾的 "dic num:XX" 条目）。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [警告] JSON 解析失败: {filepath} — {e}")
        return []

    if not isinstance(data, list):
        print(f"  [警告] 文件内容不是列表: {filepath}")
        return []

    # 过滤掉 "dic num:XX" 条目
    nodes = [item for item in data if not re.match(r"^dic num:\d+$", str(item))]
    return nodes


def build_node_dataset(graph_dir: str) -> list:
    """
    遍历 graph_dir 下的所有节点文件，构建统一的节点数据集。

    返回: list[dict], 每个 dict 格式为:
        {"name": str, "type": [str], "use": [str], "disease": [str]}
    """
    # node_map: node_name -> {"type": set(), "use": set(), "disease": set()}
    node_map = defaultdict(lambda: {"type": set(), "use": set(), "disease": set()})

    # 获取所有疾病子目录名
    disease_dirs = []
    for item in os.listdir(graph_dir):
        item_path = os.path.join(graph_dir, item)
        if os.path.isdir(item_path):
            disease_dirs.append(item)

    print(f"发现疾病子目录: {disease_dirs}")

    # ---- 第一步：处理根目录下的 type 文件 ----
    print("\n--- 处理根目录节点类型文件 ---")
    for filename in os.listdir(graph_dir):
        filepath = os.path.join(graph_dir, filename)
        if not os.path.isfile(filepath):
            continue
        if not is_node_file(filename):
            continue

        type_label = extract_type_from_filename(filename)
        if type_label is None:
            print(f"  [跳过] 无法解析类型: {filename}")
            continue

        nodes = parse_node_file(filepath)
        print(f"  {filename}: type={type_label}, 节点数={len(nodes)}")
        for node_name in nodes:
            # 按 "+" 切分复合节点（根目录文件同样可能包含组合条目，如药物组合）
            sub_names = [s.strip() for s in node_name.split("+")]
            for sub_name in sub_names:
                if not sub_name:
                    continue
                node_map[sub_name]["type"].add(type_label)

    # ---- 第二步：处理疾病子目录下的 use 文件 ----
    for disease_name in disease_dirs:
        disease_path = os.path.join(graph_dir, disease_name)
        print(f"\n--- 处理疾病子目录: {disease_name} ---")

        for filename in os.listdir(disease_path):
            filepath = os.path.join(disease_path, filename)
            if not os.path.isfile(filepath):
                continue
            if not is_node_file(filename):
                continue

            use_label = extract_type_from_filename(filename)
            if use_label is None:
                print(f"  [跳过] 无法解析用途: {filename}")
                continue

            nodes = parse_node_file(filepath)
            print(f"  {filename}: use={use_label}, disease={disease_name}, 节点数={len(nodes)}")
            for node_name in nodes:
                # 补丁：按 "+" 切分复合节点，如 "妊娠合并慢性高血压+子痫前期表现"
                # → "妊娠合并慢性高血压", "子痫前期表现"
                sub_names = [s.strip() for s in node_name.split("+")]
                for sub_name in sub_names:
                    if not sub_name:
                        continue
                    node_map[sub_name]["use"].add(use_label)
                    node_map[sub_name]["disease"].add(disease_name)

    # ---- 第三步：转换为有序列表输出 ----
    dataset = []
    for name in sorted(node_map.keys()):
        tags = node_map[name]
        dataset.append({
            "name": name,
            "type": sorted(tags["type"]),
            "use": sorted(tags["use"]),
            "disease": sorted(tags["disease"]),
        })

    return dataset


# ============================================================
# Jaccard 补丁：为 type 为空的节点从相似节点借用 type
# ============================================================

def _tokenize(name: str) -> set:
    """
    与 verify_node_text_match.py 一致的分词逻辑。
    提取连续汉字、英文字母、数字、符号作为 token。
    用于对称 Jaccard 计算时先做 normalize。
    """
    # 归一化：全角→半角、去空白、统一连字符、转小写
    text = name
    result = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            result.append(' ')
        elif 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        else:
            result.append(ch)
    text = ''.join(result)
    text = re.sub(r'\s+', '', text)
    text = text.replace('～', '~').replace('—', '-').replace('–', '-')
    text = text.replace('％', '%')
    text = text.lower()

    tokens = re.findall(
        r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+(?:\.\d+)?|[~≥≤<>±%]',
        text
    )
    # 过滤掉单字母 token（如 a, b, c, A, B），
    # 它们几乎不携带语义，却会在 Jaccard 中造成假阳性匹配
    tokens = [t for t in tokens
              if not (len(t) == 1 and t.isalpha())]
    return set(tokens)


def patch_types_by_jaccard(dataset: list, threshold: float = 0.6,
                           mode: str = "symmetric") -> int:
    """
    对于所有 use/disease 非空但 type 为空的节点 A，
    在 type 非空的节点集中通过 Jaccard 寻找最佳匹配 B，
    将 B 的 type 赋给 A。

    mode:
      "symmetric"  — 对称 Jaccard = |A ∩ B| / |A ∪ B| (更保守)
      "asymmetric" — 非对称 Jaccard = |A ∩ B| / |A|     (更激进，要求 A 的 token 在 B 中覆盖率高)

    Returns:
        成功补丁的节点数量
    """
    if mode not in ("symmetric", "asymmetric"):
        raise ValueError(f"mode 必须为 'symmetric' 或 'asymmetric'，收到: {mode}")

    # 收集候选池：type 非空的节点 (index, tokens, type_list)
    pool = []
    for idx, entry in enumerate(dataset):
        if entry["type"]:
            pool.append((idx, _tokenize(entry["name"]), entry["type"]))

    # 收集待补丁节点：type 为空 且 (use 或 disease 非空)
    targets = []
    for idx, entry in enumerate(dataset):
        if not entry["type"] and (entry["use"] or entry["disease"]):
            targets.append((idx, entry["name"], _tokenize(entry["name"])))

    if not targets:
        print("\n[补丁] 没有 type 为空且 use/disease 非空的节点，跳过。")
        return 0

    print(f"\n[补丁] 开始 Jaccard type 复用 (mode={mode}, 阈值={threshold})...")
    print(f"  候选池 (type 非空): {len(pool)} 个")
    print(f"  待补丁节点:         {len(targets)} 个")

    patched = 0
    for tgt_idx, tgt_name, tgt_tokens in targets:
        if not tgt_tokens:
            continue

        best_jaccard = 0.0
        best_type = None
        best_name = ""

        for src_idx, src_tokens, src_type in pool:
            intersection = len(tgt_tokens & src_tokens)
            if mode == "symmetric":
                union = len(tgt_tokens | src_tokens)
                if union == 0:
                    continue
                jaccard = intersection / union
            else:  # asymmetric
                jaccard = intersection / len(tgt_tokens)

            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_type = src_type
                best_name = dataset[src_idx]["name"]

        if best_jaccard >= threshold and best_type:
            dataset[tgt_idx]["type"] = best_type
            patched += 1
            if patched <= 10:
                print(f"  ✓ [{best_jaccard:.2f}] '{tgt_name[:50]}' ← '{best_name[:50]}' type={best_type}")

    print(f"  补丁完成: {patched}/{len(targets)} 个节点获得了 type")
    return patched


# ============================================================
# 规则补丁：基于子串的兜底 type 赋值
# ============================================================

def patch_types_by_rules(dataset: list) -> int:
    """
    对所有仍然 type 为空且 (use 或 disease 非空) 的节点，
    应用基于子串的规则进行兜底 type 赋值。

    规则（命中即停止）：
      1. name 含 "治疗"          → type = ["治疗方式"]
    """
    rules = [
        ("治疗", ["治疗方式"]),
    ]

    patched = 0
    for entry in dataset:
        if entry["type"] or (not entry["use"] and not entry["disease"]):
            continue

        for keyword, new_type in rules:
            if keyword in entry["name"]:
                entry["type"] = new_type
                patched += 1
                if patched <= 5:
                    print(f"  [规则] '{entry['name'][:50]}' → type={new_type}")
                break  # 命中一条规则即停止

    print(f"  规则补丁完成: {patched} 个节点获得了 type")
    return patched


def build_index(dataset: list, key: str) -> dict:
    """
    为 dataset 按指定 key 建立索引。
    返回: {value: [index_in_dataset, ...]}
    例如 key="type" → {"个人史概念": [0, 5, 12], "疾病": [3, 7], ...}
    """
    index = defaultdict(list)
    for idx, entry in enumerate(dataset):
        values = entry.get(key, [])
        for v in values:
            index[v].append(idx)
    return dict(index)


def save_index_files(index: dict, key: str, output_dir: str):
    """将单个维度的索引保存为多个 JSON 文件。"""
    for value, indices in index.items():
        # 文件名中的特殊字符处理
        safe_value = value.replace("/", "_").replace("\\", "_").replace(":", "_")
        filename = f"index_{key}_{safe_value}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(indices, f, ensure_ascii=False)


# ============================================================
# 主流程
# ============================================================

def main():
    print(f"数据目录: {GRAPH_DIR}")
    print(f"输出文件: {OUTPUT_NODE_FILE}")
    print(f"索引目录: {OUTPUT_INDEX_DIR}")

    # 构建数据集
    print("=" * 60)
    print("开始构建节点数据集...")
    print("=" * 60)
    dataset = build_node_dataset(GRAPH_DIR)
    print(f"\n总节点数: {len(dataset)}")

    # ---- 补丁：为 type 为空的节点从相似节点借用 type ----
    patched_count = patch_types_by_jaccard(dataset, threshold=JACCARD_PATCH_THRESHOLD,
                                             mode=JACCARD_PATCH_MODE)

    # ---- 规则补丁：对剩余的 type 为空节点做子串兜底 ----
    rule_patched_count = patch_types_by_rules(dataset)

    # 统计信息
    type_count = sum(1 for e in dataset if e["type"])

    print(f"  有 type 标签的节点: {type_count} (Jaccard补丁: +{patched_count}, 规则补丁: +{rule_patched_count})")
    use_count = sum(1 for e in dataset if e["use"])
    disease_count = sum(1 for e in dataset if e["disease"])
    print(f"  有 type 标签的节点: {type_count}")
    print(f"  有 use 标签的节点:  {use_count}")
    print(f"  有 disease 标签的节点: {disease_count}")

    # 保存主数据集
    os.makedirs(os.path.dirname(OUTPUT_NODE_FILE), exist_ok=True)
    with open(OUTPUT_NODE_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\n主数据集已保存: {OUTPUT_NODE_FILE}")

    # 建立并保存索引
    os.makedirs(OUTPUT_INDEX_DIR, exist_ok=True)

    for key in ["type", "use", "disease"]:
        index = build_index(dataset, key)
        save_index_files(index, key, OUTPUT_INDEX_DIR)
        print(f"索引 (key={key}) 已保存至: {OUTPUT_INDEX_DIR}/ (共 {len(index)} 个文件)")

    print("\n完成!")


if __name__ == "__main__":
    main()
