"""
sft_datasets_prompts.py
=======================
SFT 数据集构建所用的 Prompt 模板。
"""

import json

# ============================================================
# 系统级指令
# ============================================================

NODE_SFT_SYSTEM_PROMPT = (
    "你是一个医学知识图谱实体提取助手。"
    "从给定的医学文本中，识别并提取所有医学实体节点。"
    "实体类型必须严格从以下类型中选择："
    "```"
    "个人史概念、过敏史概念、家族史概念、疾病、检查检验、检查指标、检验指标、人群、体征和症状、药物药剂、治疗方式"
    "```"
    "以 JSON 格式输出，格式为 {\"nodes\": [{\"name\": \"...\", \"type\": \"...\"}, ...]}。"
)


# ============================================================
# 用户 / 助手模板
# ============================================================

NODE_SFT_USER_TEMPLATE = "请从以下医学文本中提取所有实体节点：\n\n{text}"

NODE_SFT_ASSISTANT_TEMPLATE = "```json\n{nodes_json}\n```"


# ============================================================
# 完整样本构造
# ============================================================

def build_node_sft_prompt(
    text: str,
    matched_nodes: list = None,
) -> dict:
    """
    构造一条 SFT 训练样本（节点提取任务）。

    Args:
        text: chunk 文本
        matched_nodes: 匹配到的节点列表。
                       每个元素为 {"name": str, "type": [...]}。
                       None / 空 表示负样本。

    Returns:
        {"prompt": str, "completion": str}
    """
    prompt = (
        f"<|im_start|>system\n{NODE_SFT_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{NODE_SFT_USER_TEMPLATE.format(text=text)}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if matched_nodes:
        simplified = []
        for n in matched_nodes:
            types = n.get("type", []) if isinstance(n, dict) else []
            type_str = types[0] if types else "未知"
            name = n.get("name", "") if isinstance(n, dict) else str(n)
            simplified.append({
                "name": name,
                "type": type_str,
            })

        nodes_json = json.dumps(
            {"nodes": simplified}, ensure_ascii=False, indent=2
        )
        completion = NODE_SFT_ASSISTANT_TEMPLATE.format(nodes_json=nodes_json)
    else:
        completion = "```json\n{\"nodes\": []}\n```"

    return {"prompt": prompt, "completion": completion}
