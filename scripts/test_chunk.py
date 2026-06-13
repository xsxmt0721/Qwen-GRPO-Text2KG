"""
test_chunk.py
=============
测试 text_chunker 的切分效果。
读取一份指南文本，切分为 chunk，将结果保存为 JSON。
"""

import json
import os
import sys
from pathlib import Path

# 添加项目根路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils_text import chunk_text

# ============================================================
# 配置
# ============================================================
TEXT_DIR = PROJECT_ROOT / "Data" / "text"

# 选择一份测试文档（可手动指定文件名或使用第一份糖尿病指南）
TEST_DOC = "中国2型糖尿病防治指南（2020年版）.txt"  # None = 使用第一份文档

MAX_CHUNK_LENGTH = 2000      # chunk 最大字符数
MIN_OVERLAP_LENGTH = 500     # chunk 间最小重叠字符数
MIN_STRONG_RATIO = 0.5       # 强终止符最短比例

OUTPUT_DIR = PROJECT_ROOT / "scripts" / "output"
OUTPUT_FILE = "test_chunk_output.json"


def main():
    # ---- 选择测试文档 ----
    doc_name = TEST_DOC
    if doc_name is None:
        # 使用第一份可用 txt
        txt_files = sorted(Path(TEXT_DIR).glob("*.txt"))
        if not txt_files:
            print("错误: Data/text 目录下没有 txt 文件")
            sys.exit(1)
        doc_name = txt_files[0].name

    filepath = TEXT_DIR / doc_name
    if not filepath.exists():
        print(f"错误: 文件不存在 {filepath}")
        # 列出可用文件
        print("\n可用的测试文档:")
        for f in sorted(Path(TEXT_DIR).glob("*.txt")):
            print(f"  {f.name}")
        sys.exit(1)

    print(f"测试文档: {doc_name}")

    # ---- 读取文本 ----
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"文本长度: {len(text):,} 字符")
    print(f"参数: max_chunk_length={MAX_CHUNK_LENGTH}, "
          f"min_overlap_length={MIN_OVERLAP_LENGTH}, "
          f"min_strong_ratio={MIN_STRONG_RATIO}")

    # ---- 切分 ----
    chunks = chunk_text(
        text,
        max_chunk_length=MAX_CHUNK_LENGTH,
        min_overlap_length=MIN_OVERLAP_LENGTH,
        min_strong_ratio=MIN_STRONG_RATIO,
    )

    # ---- 统计 ----
    lengths = [len(c) for c in chunks]
    print(f"\n切分结果:")
    print(f"  chunk 数量: {len(chunks)}")
    print(f"  平均长度:   {sum(lengths) / max(len(lengths), 1):.0f} 字符")
    print(f"  最短长度:   {min(lengths) if lengths else 0} 字符")
    print(f"  最长长度:   {max(lengths) if lengths else 0} 字符")
    if lengths:
        over_limit = [l for l in lengths if l > MAX_CHUNK_LENGTH]
        if over_limit:
            print(f"  ⚠ 超限 chunk: {len(over_limit)} 个 "
                  f"(max={max(over_limit)})")
        else:
            print(f"  ✅ 所有 chunk 均 ≤ {MAX_CHUNK_LENGTH} 字符")

    # ---- 展示样例 ----
    print(f"\n前 5 个 chunk 预览:")
    for i, chunk in enumerate(chunks[:5]):
        preview = chunk[:80].replace('\n', '↵')
        suffix = "..." if len(chunk) > 80 else ""
        print(f"\n  --- Chunk {i} ({len(chunk)} 字符) ---")
        print(f"  首位: {preview}{suffix}")
        end_preview = chunk[-40:].replace('\n', '↵')
        print(f"  末位: ...{end_preview}")

    # ---- 保存结果 ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILE

    result = {
        "doc_name": doc_name,
        "text_length": len(text),
        "parameters": {
            "max_chunk_length": MAX_CHUNK_LENGTH,
            "min_overlap_length": MIN_OVERLAP_LENGTH,
            "min_strong_ratio": MIN_STRONG_RATIO,
        },
        "num_chunks": len(chunks),
        "chunk_lengths": lengths,
        "avg_length": sum(lengths) / max(len(lengths), 1),
        "chunks": [
            {
                "index": i,
                "length": len(c),
                "start_preview": c[:60].replace('\n', '↵'),
                "end_preview": c[-40:].replace('\n', '↵'),
                "text": c,  # 完整 chunk 文本
            }
            for i, c in enumerate(chunks)
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_path}")
    print("完成!")


if __name__ == "__main__":
    main()
