"""
verify_node_text_match.py
==========================
确认 node.json 中的节点名称在指南原文中的字符串匹配度。

运行方式（容器内）：
  python scripts/verify_node_text_match.py

输出：
  1. 终端打印匹配统计（按匹配层级分组）
  2. scripts/output/match_report.json — 完整报告
  3. scripts/output/unmatched_nodes.txt — 未匹配节点列表（供后续处理用）

匹配层级（从严格到宽松）：
  Level 0: 原始精确子串匹配            node_name in text
  Level 1: 归一化后子串匹配            规范化空格/全角半角/标点后匹配
  Level 2: 分词 Jaccard 模糊匹配       token-level overlap >= threshold
  Level 3: 同文档跨 chunk 拆词匹配      节点拆分后在同文档中所有 piece 均出现
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set, Tuple
from tqdm import tqdm

# 文本分段工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.utils_text import chunk_text


# ============================================================
# 路径配置（容器内）
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # scripts/ → 项目根

NODE_FILE = PROJECT_ROOT / "Data" / "data" / "node.json"
TEXT_DIR  = PROJECT_ROOT / "Data" / "text"
INDEX_DIR = PROJECT_ROOT / "Data" / "data" / "index"

OUTPUT_DIR = PROJECT_ROOT / "scripts" / "output"

# ============================================================
# Chunk 参数
# ============================================================
MAX_CHUNK_LENGTH = 500       # chunk 最大字符数
MIN_OVERLAP_LENGTH = 200     # chunk 间最小重叠字符数


# ============================================================
# 字符串归一化
# ============================================================

def normalize(text: str) -> str:
    """
    统一规范化以便匹配：
    - 全角→半角
    - 去除所有空白字符（包括 Unicode 空白）
    - 统一破折号/连字符
    - 转小写
    """
    # 全角→半角
    result = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:       # 全角空格 → 半角空格
            result.append(' ')
        elif 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif 0xFF65 <= code <= 0xFF9F:  # 半角片假名范围跳过
            result.append(ch)
        else:
            result.append(ch)
    text = ''.join(result)

    # 去除所有空白字符（包括 \u200b \u00a0 等）
    text = re.sub(r'\s+', '', text)

    # 统一连字符
    text = text.replace('～', '~').replace('—', '-').replace('–', '-')

    # 统一百分号
    text = text.replace('％', '%')

    return text.lower()


def tokenize(text: str) -> Set[str]:
    """
    中文+英文混合分词。
    提取连续的汉字串、英文字母串、数字串作为 token。
    """
    tokens = re.findall(
        r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+(?:\.\d+)?|[~≥≤<>±%]',
        normalize(text)
    )
    return set(tokens)


def split_node_to_pieces(node_name: str) -> List[str]:
    """
    将节点名拆分为有意义的片段。
    例如 "0.5h血糖10.0~11.1mmol/L" → ["0.5h", "血糖", "10.0~11.1", "mmol/L"]
    """
    # 按常见的组合边界拆分
    pieces = re.findall(
        r'[\u4e00-\u9fff]+|[a-zA-Z/]+|\d+(?:\.\d+)?[~-]\d+(?:\.\d+)?|\d+(?:\.\d+)?',
        node_name
    )
    return [p for p in pieces if len(p) >= 2]


# ============================================================
# 数据加载
# ============================================================

def load_nodes(filepath: Path) -> List[dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_texts(text_dir: Path) -> Dict[str, str]:
    """加载所有 txt 文件。key=文档名(不含.txt), value=全文"""
    docs = {}
    for filepath in sorted(text_dir.glob("*.txt")):
        doc_name = filepath.stem
        with open(filepath, 'r', encoding='utf-8') as f:
            docs[doc_name] = f.read()
    return docs


def load_index(filepath: Path) -> List[int]:
    """加载索引文件（node ID 列表）"""
    if not filepath.exists():
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# 匹配函数
# ============================================================

def match_level0(node_name: str, text: str) -> bool:
    """Level 0: 原始字符串精确子串匹配"""
    return node_name in text


def match_level1(node_name: str, text: str) -> bool:
    """Level 1: 归一化后子串匹配"""
    return normalize(node_name) in normalize(text)


def match_level1_5(node_name: str, text: str) -> bool:
    """
    Level 1.5: 约束型数值范围校验 + 上下文 Jaccard。
    与 utils_text._match_level1_5 逻辑一致（简化版，仍返回 bool）。
    """
    # 提取数值范围
    ranges = re.findall(r'(\d+(?:\.\d+)?)\s*[~～-]\s*(\d+(?:\.\d+)?)', node_name)
    if not ranges:
        return False

    norm_text = normalize(text)

    # 剥离节点中的数字和连接符 token
    node_tokens_full = tokenize(node_name)
    stripped = {t for t in node_tokens_full
                if re.match(r'^\d+(?:\.\d+)?$', t) or t == '~'}
    node_tokens_A = node_tokens_full - stripped

    # 对每个范围: 查找→取上下文→增强 Jaccard
    for lo, hi in ranges:
        pattern = re.escape(lo) + r'\s*[~～-]\s*' + re.escape(hi)
        m = re.search(pattern, norm_text)
        if not m:
            continue

        start = max(0, m.start() - 30)
        end = min(len(norm_text), m.end() + 30)
        context_tokens = tokenize(norm_text[start:end])
        if not context_tokens:
            continue

        intersection = len(node_tokens_A & context_tokens)
        score = (intersection + 1) / (len(node_tokens_A) + 1)
        if score >= 0.6:
            return True

    return False


def match_level2_in_chunk(node_name: str, chunk_text: str, threshold: float = 0.6) -> bool:
    """
    Level 2: 分词 Jaccard 模糊匹配（单 chunk 版本）。
    节点 token 集与单个 chunk 文本 token 集的重叠度。
    """
    node_tokens = tokenize(node_name)
    if not node_tokens:
        return False
    chunk_tokens = tokenize(chunk_text)
    overlap = len(node_tokens & chunk_tokens)
    jaccard = overlap / len(node_tokens)
    return jaccard >= threshold


# ---- 单 chunk 匹配入口 ----

def _match_node_in_chunks(
    node_name: str,
    chunks: List[str],
) -> str:
    """
    在 chunk 列表中逐 chunk 匹配节点名。
    逐级尝试 L0 → L1 → L1.5 → L2 → L3，命中即返回匹配层级名。
    未命中返回 "unmatched"。
    """
    for chunk_text in chunks:
        try:
            if match_level0(node_name, chunk_text):
                return "level0"
            if match_level1(node_name, chunk_text):
                return "level1"
            if match_level1_5(node_name, chunk_text):
                return "level1_5"
            if match_level2(node_name, chunk_text):
                return "level2"
            if match_level3(node_name, chunk_text):
                return "level3"
        except Exception:
            continue
    return "unmatched"


def match_level3(node_name: str, text: str) -> bool:
    """
    Level 3: 跨 chunk 拆词匹配。
    将节点拆分为片段，检查是否所有片段都在文本中出现。
    适用于 "2型糖尿病防治指南" 这类可能被换行打断的情况。
    """
    pieces = split_node_to_pieces(node_name)
    if len(pieces) <= 1:
        return False  # 无法拆分，交给上层处理
    norm_text = normalize(text)
    return all(normalize(p) in norm_text for p in pieces)


# ============================================================
# 主分析逻辑
# ============================================================

def analyze(nodes: List[dict], docs: Dict[str, str],
            disease_indexes: Dict[str, List[int]]) -> dict:
    """
    对每个节点尝试在"该节点所属疾病的文档集合"中进行匹配。

    匹配策略：
    1. 先确定节点可能关联的疾病（通过 index 文件）
    2. 在该疾病的文档集合中搜索
    3. 如果节点没有疾病标签（disease=[]），则在全部文档中搜索
    4. 逐级尝试 Level0 → Level1 → Level2 → Level3，命中即停止

    Returns:
        {
            "summary": {...},
            "by_level": {"level0": count, ...},
            "by_type": {"检验指标": {"matched": ..., "total": ...}, ...},
            "by_disease": {...},
            "unmatched": [{"id": int, "name": str, "type": [...], "disease": [...]}, ...],
            "matched_examples": [...],
            "per_node_details": [...]   # 每个节点的匹配详情
        }
    """
    # 构建疾病→文档名映射
    DISEASE_DOC_MAP = {
        "糖尿病": [],
        "高血压": [],
    }
    for doc_name in docs:
        lower = doc_name.lower()
        if "糖尿病" in lower:
            DISEASE_DOC_MAP["糖尿病"].append(doc_name)
        if "高血压" in lower:
            DISEASE_DOC_MAP["高血压"].append(doc_name)

    # 确保每个文档至少归属一个疾病
    for doc_name in docs:
        has_disease = any(
            doc_name in docs_list
            for docs_list in DISEASE_DOC_MAP.values()
        )
        if not has_disease:
            # 默认归入全部疾病
            for disease in DISEASE_DOC_MAP:
                DISEASE_DOC_MAP[disease].append(doc_name)

    print(f"文档分布:")
    for disease, doc_names in DISEASE_DOC_MAP.items():
        print(f"  {disease}: {len(doc_names)} 个文档")

    # 统计容器
    by_level = {"level0": 0, "level1": 0, "level1_5": 0, "level2": 0, "level3": 0,
                 "unmatched": 0, "ignored": 0}
    by_type: Dict[str, dict] = defaultdict(lambda: {"matched": 0, "total": 0})
    by_disease: Dict[str, dict] = defaultdict(lambda: {"matched": 0, "total": 0})
    unmatched = []
    matched_examples = []  # 每层保存少量示例
    per_node_details = []

    max_examples_per_level = 10
    level_example_counts = defaultdict(int)

    total_nodes = len(nodes)

    for idx, node in tqdm(enumerate(nodes), total=total_nodes, desc="  匹配节点"):
        node_name = node["name"]
        node_types = node.get("type", [])
        node_diseases = node.get("disease", [])

        # 按类型计数
        for t in node_types:
            by_type[t]["total"] += 1

        # 确定搜索范围：优先在节点关联疾病的文档中搜索
        search_docs = set()
        for d in node_diseases:
            if d in DISEASE_DOC_MAP:
                search_docs.update(DISEASE_DOC_MAP[d])

        # 如果节点没有疾病标签，在全部文档中搜索
        if not search_docs:
            search_docs = set(docs.keys())
            search_disease = "通用"
        else:
            search_disease = "+".join(node_diseases)

        # 拼接搜索文本后切分为 chunk
        combined_text = "\n".join(
            docs[d] for d in search_docs if d in docs
        )
        chunks = chunk_text(
            combined_text,
            max_chunk_length=MAX_CHUNK_LENGTH,
            min_overlap_length=MIN_OVERLAP_LENGTH,
        )

        # 逐 chunk 匹配
        match_level = _match_node_in_chunks(node_name, chunks)

        # 记录结果
        by_level[match_level] += 1
        matched = (match_level != "unmatched")

        detail = {
            "id": idx,
            "name": node_name,
            "types": node_types,
            "diseases": node_diseases,
            "match_level": match_level,
            "search_disease": search_disease,
        }
        per_node_details.append(detail)

        if matched:
            for t in node_types:
                by_type[t]["matched"] += 1
            if node_diseases:
                for d in node_diseases:
                    by_disease[d]["matched"] += 1
            else:
                # 无疾病标签的节点 → 计入 "通用"
                by_disease["通用"]["matched"] += 1

            # 收集示例
            level_key = match_level
            if level_example_counts[level_key] < max_examples_per_level:
                matched_examples.append({
                    "node_name": node_name,
                    "types": node_types,
                    "diseases": node_diseases,
                    "match_level": match_level,
                })
                level_example_counts[level_key] += 1
        else:
            # 未匹配节点的 total 交给底部补充循环统一统计，这里只收集列表
            unmatched.append({
                "id": idx,
                "name": node_name,
                "types": node_types,
                "diseases": node_diseases,
            })

    # 补充 by_disease 的 total（统一统计，避免双重计数）
    for node in nodes:
        diseases = node.get("disease", [])
        if diseases:
            for d in diseases:
                if d not in by_disease:
                    by_disease[d] = {"matched": 0, "total": 0}
                by_disease[d]["total"] += 1
        else:
            if "通用" not in by_disease:
                by_disease["通用"] = {"matched": 0, "total": 0}
            by_disease["通用"]["total"] += 1

    summary = {
        "total_nodes": total_nodes,
        "total_documents": len(docs),
        "match_rate": round(
            (total_nodes - by_level["unmatched"]) / total_nodes * 100, 2
        ),
    }

    return {
        "summary": summary,
        "by_level": dict(by_level),
        "by_type": {k: dict(v) for k, v in by_type.items()},
        "by_disease": {k: dict(v) for k, v in by_disease.items()},
        "unmatched": unmatched,
        "matched_examples": matched_examples,
        "per_node_details": per_node_details,
    }


# ============================================================
# 按疾病索引过滤的分析（窄范围，更高精度）
# ============================================================

def analyze_with_disease_index(
    nodes: List[dict],
    docs: Dict[str, str],
    disease_indexes: Dict[str, List[int]],
) -> dict:
    """
    使用 index 文件精确定位每个节点的疾病归属，
    只在对应疾病的文本中搜索。
    同时统计"有疾病标签的节点"的匹配率（更真实的训练数据指标）。
    """
    print("\n" + "=" * 60)
    print("进阶分析：按疾病索引缩小搜索范围")
    print("=" * 60)

    DISEASE_DOC_MAP = {
        "糖尿病": [d for d in docs if "糖尿病" in d],
        "高血压": [d for d in docs if "高血压" in d],
    }

    results = {}
    for disease in ["糖尿病", "高血压"]:
        node_ids = set(disease_indexes.get(disease, []))
        if not node_ids:
            print(f"  {disease}: 无索引数据，跳过")
            continue

        disease_docs = DISEASE_DOC_MAP.get(disease, list(docs.keys()))
        combined_text = "\n".join(docs[d] for d in disease_docs if d in docs)
        chunks = chunk_text(
            combined_text,
            max_chunk_length=MAX_CHUNK_LENGTH,
            min_overlap_length=MIN_OVERLAP_LENGTH,
        )

        matched_count = {f"level{i}": 0 for i in range(4)}
        matched_count["level1_5"] = 0
        matched_count["unmatched"] = 0
        total = 0
        unmatched_names = []

        for idx in tqdm(sorted(node_ids), desc=f"  {disease} 节点"):
            if idx >= len(nodes):
                continue
            total += 1
            node = nodes[idx]
            node_name = node["name"]

            level = _match_node_in_chunks(node_name, chunks)
            matched_count[level] += 1
            if level == "unmatched" and len(unmatched_names) < 20:
                unmatched_names.append(node_name)

        match_rate = round(
            (total - matched_count["unmatched"]) / max(total, 1) * 100, 2
        )

        results[disease] = {
            "total": total,
            "matched": total - matched_count["unmatched"],
            "match_rate": match_rate,
            "by_level": dict(matched_count),
            "sample_unmatched": unmatched_names,
            "docs_used": disease_docs,
        }

        print(f"\n  {disease}:")
        print(f"    节点总数: {total}")
        print(f"    搜索文档: {len(disease_docs)} 个")
        print(f"    匹配率: {match_rate}%")
        for level, count in matched_count.items():
            if count > 0:
                print(f"      {level}: {count} ({round(count/max(total,1)*100,1)}%)")
        if unmatched_names:
            print(f"    未匹配样例: {unmatched_names[:5]}")

    return results


# ============================================================
# 输出
# ============================================================

def print_report(report: dict, advanced_report: dict = None):
    """打印匹配报告到终端"""
    summary = report["summary"]
    by_level = report["by_level"]
    by_type = report["by_type"]

    print("\n" + "=" * 60)
    print("节点-文本 字符串匹配验证报告")
    print("=" * 60)

    print(f"\n数据概览:")
    print(f"  节点总数:     {summary['total_nodes']}")
    print(f"  文档总数:     {summary['total_documents']}")
    print(f"  总体匹配率:   {summary['match_rate']}%")

    print(f"\n按匹配层级:")
    level_names = {
        "level0": "L0 原始精确子串",
        "level1": "L1 归一化后子串",
        "level1_5": "L1.5 数值范围校验",
        "level2": "L2 分词Jaccard",
        "level3": "L3 拆词匹配",
        "unmatched": "✗ 未匹配",
        "ignored": "⊘ 忽略",
    }
    for level, count in by_level.items():
        label = level_names.get(level, level)
        pct = round(count / max(summary["total_nodes"], 1) * 100, 1)
        bar = "█" * int(pct / 2)
        print(f"  {label:20s}: {count:>5} ({pct:>5.1f}%) {bar}")

    # 按类型
    print(f"\n按节点类型:")
    for t, stats in sorted(by_type.items(),
                            key=lambda x: x[1]["total"], reverse=True):
        total = stats["total"]
        matched = stats.get("matched", 0)
        rate = round(matched / max(total, 1) * 100, 1)
        print(f"  {t:20s}: {matched:>4}/{total:<4} ({rate:>5.1f}%)")

    # 匹配示例
    print(f"\n匹配示例（每层最多10条）:")
    for ex in report["matched_examples"][:30]:
        print(f"  [{ex['match_level']}] {ex['node_name'][:60]}")

    # 未匹配节点数
    unmatched = report.get("unmatched", [])
    print(f"\n未匹配节点: {len(unmatched)} 个")
    if unmatched:
        print(f"  前10个未匹配节点:")
        for u in unmatched[:10]:
            print(f"    [{u['id']}] {u['name']}")

    # 结论
    print(f"\n{'=' * 60}")
    print(f"结论与建议:")
    l0_rate = by_level.get("level0", 0) / max(summary["total_nodes"], 1) * 100
    l01_rate = (by_level.get("level0", 0) + by_level.get("level1", 0)) / max(summary["total_nodes"], 1) * 100
    l015_rate = (by_level.get("level0", 0) + by_level.get("level1", 0) + by_level.get("level1_5", 0)) / max(summary["total_nodes"], 1) * 100
    l012_rate = (by_level.get("level0", 0) + by_level.get("level1", 0) + by_level.get("level1_5", 0) + by_level.get("level2", 0)) / max(summary["total_nodes"], 1) * 100

    print(f"  L0 精确匹配率: {l0_rate:.1f}%")
    print(f"  L0+L1 归一化匹配率: {l01_rate:.1f}%")
    print(f"  L0+L1+L1.5 数值校验匹配率: {l015_rate:.1f}%")
    print(f"  L0+L1+L1.5+L2 综合匹配率: {l012_rate:.1f}%")

    if l01_rate >= 85:
        print(f"  ✅ L0+L1 已覆盖绝大多数节点，仅需处理少量未匹配节点")
        print(f"     建议：对未匹配节点使用 BERT 语义匹配或 DeepSeek API")
    elif l01_rate >= 65:
        print(f"  ⚠️  L0+L1 覆盖中等，L2 可补充部分")
        print(f"     建议：L0+L1+L2 综合使用，剩余未匹配节点跑 BERT")
    else:
        print(f"  ❌ 字符串匹配覆盖率偏低，建议考虑 BERT 语义匹配全量使用")

    print(f"{'=' * 60}\n")


def save_report(report: dict, advanced_report: dict = None):
    """保存完整报告到 JSON 文件"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 完整报告（不含 per_node_details 避免过大）
    full_report = {
        "summary": report["summary"],
        "by_level": report["by_level"],
        "by_type": report["by_type"],
        "by_disease": report["by_disease"],
        "unmatched_count": len(report["unmatched"]),
        "matched_examples": report["matched_examples"],
    }

    if advanced_report:
        full_report["advanced_by_disease"] = advanced_report

    report_path = OUTPUT_DIR / "match_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    print(f"完整报告: {report_path}")

    # 未匹配节点列表（供后续 BERT/API 处理用）
    unmatched_path = OUTPUT_DIR / "unmatched_nodes.json"
    with open(unmatched_path, 'w', encoding='utf-8') as f:
        json.dump(report["unmatched"], f, ensure_ascii=False, indent=2)
    print(f"未匹配节点: {unmatched_path} ({len(report['unmatched'])} 个)")

    # 人类可读的未匹配列表
    txt_path = OUTPUT_DIR / "unmatched_nodes.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"未匹配节点列表（共 {len(report['unmatched'])} 个）\n")
        f.write("=" * 60 + "\n\n")
        for u in report["unmatched"]:
            f.write(f"[{u['id']}] {u['name']}\n")
            f.write(f"    类型: {u['types']}\n")
            f.write(f"    疾病: {u['diseases']}\n\n")
    print(f"未匹配详情: {txt_path}")


# ============================================================
# 主入口
# ============================================================

def main():
    print("=" * 60)
    print("节点-文本 字符串匹配验证")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/4] 加载节点数据...")
    nodes = load_nodes(NODE_FILE)
    print(f"  加载 {len(nodes)} 个节点")

    print("\n[2/4] 加载文本数据...")
    docs = load_texts(TEXT_DIR)
    total_chars = sum(len(t) for t in docs.values())
    print(f"  加载 {len(docs)} 个文档，共 {total_chars:,} 字符")

    print("\n[3/4] 加载疾病索引...")
    disease_indexes = {}
    for disease in ["糖尿病", "高血压"]:
        idx_path = INDEX_DIR / f"index_disease_{disease}.json"
        ids = load_index(idx_path)
        disease_indexes[disease] = ids
        print(f"  {disease}: {len(ids)} 个节点索引")

    # 2. 全量匹配分析
    print("\n[4/4] 执行字符串匹配分析...")
    print("  （此过程可能需要 1-2 分钟，涉及数千节点 × 百万字符文本）")
    report = analyze(nodes, docs, disease_indexes)

    # 3. 进阶分析（按疾病索引缩小范围）
    advanced_report = analyze_with_disease_index(nodes, docs, disease_indexes)

    # 4. 输出
    print_report(report, advanced_report)
    save_report(report, advanced_report)

    # 5. 返回未匹配节点数量（供脚本调用）
    return len(report["unmatched"])


if __name__ == "__main__":
    unmatched_count = main()
    sys.exit(0 if unmatched_count == 0 else 1)
