"""
utils_text.py
=============
文本分段工具。将长文本按约束切分为适合 LLM 训练的 chunk 列表。
"""

import re
from typing import List


# ============================================================
# 标点模式
# ============================================================

# 强终止符：句号、分号、问号、感叹号（中英文）、换行
_STRONG_BREAK = re.compile(r'[。；？?！!\n]')

# 弱终止符：逗号、顿号、右括号（中英文）、冒号
_WEAK_BREAK = re.compile(r'[，,、）\)：:]')

# 强 + 弱 = 有效的句子分隔符（用于判断 chunk 起始位置合法性）
_ANY_BREAK_CHARS = set('。；？?！!；;，,、）\)：:\n\r\t ')


# ============================================================
# 辅助函数
# ============================================================

def _is_valid_chunk_start(text: str, pos: int) -> bool:
    """pos 处是否是一个合法的 chunk 起始位置。"""
    if pos == 0:
        return True
    if pos >= len(text):
        return False
    prev = text[pos - 1]
    return prev in _ANY_BREAK_CHARS or prev.isspace()


def _find_last_valid_start(text: str, lo: int, hi: int) -> int:
    """
    在 (lo, hi] 范围内，找到最大的满足 _is_valid_chunk_start 的位置。
    即尽可能靠近 hi（=cut-min_overlap），最大化 chunk 重叠利用率。
    如无，返回 -1。
    """
    for i in range(hi, lo, -1):
        if _is_valid_chunk_start(text, i):
            return i
    return -1


# ============================================================
# 主函数
# ============================================================

def chunk_text(
    text: str,
    max_chunk_length: int,
    min_overlap_length: int = 0,
    min_strong_ratio: float = 0.0,   # 保留签名兼容，本版不使用
) -> List[str]:
    """
    将长文本贪心切分为 chunk 列表。

    规则（按优先级）：
      1. 长度约束:  每个 chunk 长度 ≤ max_chunk_length
      2. 结尾约束:  尽量以强终止符（。；？！\n）结尾；
                    无强终止符时回退到弱终止符（，、）:）；
                    均无则硬截断。
      3. 开头约束:  每个 chunk 的起始位置必须是强/弱终止符的下一个字符。
      4. 重叠约束:  下一个 chunk 的起始位置必须使得 chunk 间存在
                    至少 min_overlap_length 字符的连续重叠区域。
      5. 贪心目标:  在上述约束下，每个 chunk 尽可能长（从右端选取切分点）。

    Args:
        text:              待切分的全文
        max_chunk_length:  每个 chunk 的最大字符数
        min_overlap_length: chunk 间最小重叠字符数（0 = 不重叠）
        min_strong_ratio:  保留参数，本版未使用。

    Returns:
        有序 chunk 字符串列表
    """
    if not text:
        return []

    n = len(text)
    chunks: List[str] = []

    pos = 0
    while pos < n:
        # ============================================================
        # Step 1: 确定搜索窗口右边界（硬上限）
        # ============================================================
        hard_end = min(pos + max_chunk_length, n)

        # ============================================================
        # Step 2: 在 [pos, hard_end) 内找最优切分点
        # ============================================================
        window = text[pos:hard_end]

        # 找最后一个强终止符
        strong_matches = list(_STRONG_BREAK.finditer(window))
        last_strong = pos + strong_matches[-1].end() if strong_matches else None

        # 找最后一个弱终止符
        weak_matches = list(_WEAK_BREAK.finditer(window))
        last_weak = pos + weak_matches[-1].end() if weak_matches else None

        # 贪心决策：取最靠后的终止符位置
        if last_strong is not None:
            cut = last_strong
        elif last_weak is not None:
            cut = last_weak
        else:
            cut = hard_end  # 硬截断

        # ============================================================
        # Step 3: 提取 chunk
        # ============================================================
        chunk = text[pos:cut].strip()
        if chunk:
            chunks.append(chunk)

        if cut >= n:
            break

        # ============================================================
        # Step 4: 决定下一个 chunk 起始位置（满足 overlap + 开头约束）
        # ============================================================
        if min_overlap_length > 0:
            # 下一个 chunk 起始位置的搜索上限 = cut - min_overlap_length
            # 即必须保留至少 min_overlap_length 的重叠字符
            overlap_hi = max(pos + 1, cut - min_overlap_length)

            # 在 (pos, overlap_hi] 内找最后一个合法的 chunk 起始位置
            next_pos = _find_last_valid_start(text, pos, overlap_hi)

            if next_pos != -1 and next_pos < cut:
                pos = next_pos
                continue

        # 无 overlap 要求或无法满足 overlap + 开头约束 → 直接前进
        pos = cut
        while pos < n and text[pos] in ' \n\r\t':
            pos += 1

    return chunks


# ============================================================
# Chunk 数据清洗
# ============================================================

# 内置过滤规则列表，每条规则是一个 (正则, 替换文本, 描述) 三元组
_CHUNK_FILTER_RULES: list = [
    # 去除 Markdown 图片链接: ![...](...)
    (re.compile(r'!\[[^\]]*\]\([^\)]+\)'), '', 'markdown_image'),
    # 去除 HTML img 标签
    (re.compile(r'<img[^>]+/?>', re.IGNORECASE), '', 'html_image'),
    # 去除 base64 编码的图片数据
    (re.compile(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+'), '', 'base64_image'),
    # 去除多余空白行（3 个以上连续换行压缩为 2 个）
    (re.compile(r'\n{3,}'), '\n\n', 'excess_newlines'),
]


def chunk_filter(chunk: str, extra_rules: list = None) -> str:
    """
    对单个 chunk 文本执行数据清洗。

    内置规则：
      - 去除 Markdown 图片链接  ![...](...)
      - 去除 HTML <img> 标签
      - 去除 base64 编码图片数据
      - 压缩多余空白行

    可通过 extra_rules 传入额外规则，格式与 _CHUNK_FILTER_RULES 一致：
      [(compiled_regex, replacement, rule_name), ...]

    Args:
        chunk: 原始 chunk 文本
        extra_rules: 额外的过滤规则列表

    Returns:
        清洗后的文本
    """
    rules = list(_CHUNK_FILTER_RULES)
    if extra_rules:
        rules.extend(extra_rules)

    for pattern, replacement, _rule_name in rules:
        chunk = pattern.sub(replacement, chunk)

    return chunk.strip()


# ============================================================
# Chunk 节点匹配（与 verify_node_text_match 一致的 L0-L2）
# ============================================================

# ---- 匹配函数（本地副本，避免跨文件依赖） ----

def _normalize(text: str) -> str:
    """全角→半角、去空白、统一连字符、转小写"""
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
    return text.lower()


def _tokenize(text: str) -> set:
    """中文+英文混合分词"""
    tokens = re.findall(
        r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+(?:\.\d+)?|[~≥≤<>±%]',
        _normalize(text)
    )
    return set(tokens)


def _match_level0(node_name: str, text: str) -> bool:
    """L0: 原始字符串精确子串匹配"""
    return node_name in text


def _match_level1(node_name: str, text: str) -> bool:
    """L1: 归一化后子串匹配"""
    return _normalize(node_name) in _normalize(text)


def _match_level1_5(
    node_name: str,
    text: str,
    threshold: float = 0.6,
    mode: str = "asymmetric",
    context_window: int = 30,
) -> tuple:
    """
    L1.5: 约束型数值范围校验 + 上下文 Jaccard。

    1. 提取节点名中的数值范围（如 "14.0~15.0"），在文本中精确搜索。
    2. 对每个匹配位置，以它为中心取 ±context_window 的上下文窗口。
    3. 从节点 token 集中剥离纯数字和连接符 token（它们在 L2 中会被拆散），
       并将数值范围整体作为一个"奖励 token"。
    4. 在上下文窗口上计算增强 Jaccard：
          score = (|A ∩ B| + 1) / (|A| + 1)    （asymmetric）
          score = (|A ∩ B| + 1) / (|A ∪ B| + 1)（symmetric）
       其中 +1 代表数值范围整体命中。

    Returns:
        (matched: bool, score: float)
    """
    # Step 1: 提取数值范围
    ranges = re.findall(r'(\d+(?:\.\d+)?)\s*[~～-]\s*(\d+(?:\.\d+)?)', node_name)
    if not ranges:
        return False, 0.0

    norm_text = _normalize(text)

    # Step 2: 剥离节点中的"纯数字"token 和"连接符"token — 它们将被替换为整体范围 token
    node_tokens_full = _tokenize(node_name)
    # 需要剥去的 token: 纯数字 (如 "14.0", "3") + 连接符 ("~")
    stripped_tokens = set()
    for token in node_tokens_full:
        # 纯数字（含小数点）
        if re.match(r'^\d+(?:\.\d+)?$', token):
            stripped_tokens.add(token)
        # 连接符
        if token == '~':
            stripped_tokens.add(token)

    node_tokens_A = node_tokens_full - stripped_tokens  # A: 去数值/连接符后的 token 集

    # Step 3: 对每个数值范围，在文本中找到匹配位置，做上下文 Jaccard
    for lo, hi in ranges:
        pattern = re.escape(lo) + r'\s*[~～-]\s*' + re.escape(hi)
        m = re.search(pattern, norm_text)
        if not m:
            continue

        # 取上下文窗口
        start = max(0, m.start() - context_window)
        end = min(len(norm_text), m.end() + context_window)
        context = norm_text[start:end]

        context_tokens_B = _tokenize(context)
        if not context_tokens_B:
            continue

        intersection = len(node_tokens_A & context_tokens_B)

        if mode == "symmetric":
            union = len(node_tokens_A | context_tokens_B)
            score = (intersection + 1) / (union + 1) if union > 0 else 0.0
        else:
            # asymmetric: 分母 = |A| + 1
            score = (intersection + 1) / (len(node_tokens_A) + 1)

        if score >= threshold:
            return True, round(score, 4)

    return False, 0.0


def _match_level2(node_name: str, text: str, threshold: float = 0.6,
                  mode: str = "asymmetric") -> tuple:
    """
    L2: Jaccard 模糊匹配。
    Returns: (matched: bool, score: float)
    """
    node_tokens = _tokenize(node_name)
    if not node_tokens:
        return False, 0.0
    text_tokens = _tokenize(text)
    intersection = len(node_tokens & text_tokens)
    if mode == "symmetric":
        union = len(node_tokens | text_tokens)
        score = intersection / union if union > 0 else 0.0
    else:
        score = intersection / len(node_tokens)
    return score >= threshold, score


# ---- 主匹配入口 ----

def chunk_node_match(
    chunk_text: str,
    nodes: list,            # list of dicts with "name", "id", "type" keys
    mode: str = "asymmetric",
    threshold: float = 0.6,
    use_l1_5: bool = True,
    l1_5_context_window: int = 30,
) -> list:
    """
    对单个 chunk 匹配节点列表中的所有节点。

    匹配顺序: L0 → L1 → L1.5(可选) → L2
    命中即停止，每个节点至多匹配一次。

    Args:
        chunk_text: 单个 chunk 的文本
        nodes: 节点列表，每个元素为 {"name": str, "id": int, "type": [str]}
        mode: Jaccard 模式 "symmetric" | "asymmetric"
        threshold: L2 Jaccard 阈值
        use_l1_5: 是否启用 L1.5 数值范围校验

    Returns:
        匹配到的节点信息列表:
        [
            {
                "name": str,       # 节点名称
                "id": int,         # 节点在 node.json 中的索引
                "type": [str],     # 节点类型列表（直接来自 node.json）
                "level": str,      # "l0" | "l1" | "l1_5" | "l2"
                "score": float,    # 匹配得分 (l0/l1/l1.5=1.0, l2=实际Jaccard)
            },
            ...
        ]
    """
    matched = []

    for node in nodes:
        node_name = node["name"]
        node_id = node.get("id", -1)
        node_type = node.get("type", [])        # ← 直接从 nodes 参数拿 type

        # L0
        if _match_level0(node_name, chunk_text):
            matched.append({
                "name": node_name,
                "id": node_id,
                "type": node_type,
                "level": "l0",
                "score": 1.0,
            })
            continue

        # L1
        if _match_level1(node_name, chunk_text):
            matched.append({
                "name": node_name,
                "id": node_id,
                "type": node_type,
                "level": "l1",
                "score": 1.0,
            })
            continue

        # L1.5 (可选)
        if use_l1_5:
            ok, score = _match_level1_5(
                node_name, chunk_text,
                threshold=threshold, mode=mode,
                context_window=l1_5_context_window,
            )
            if ok:
                matched.append({
                    "name": node_name,
                    "id": node_id,
                    "type": node_type,
                    "level": "l1_5",
                    "score": score,
                })
                continue

        # L2
        ok, score = _match_level2(node_name, chunk_text, threshold, mode)
        if ok:
            matched.append({
                "name": node_name,
                "id": node_id,
                "type": node_type,
                "level": "l2",
                "score": round(score, 4),
            })

    return matched

