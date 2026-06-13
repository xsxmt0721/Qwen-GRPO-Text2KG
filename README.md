### scripts 使用指南
- `getNodesGroup`：对源数据的节点进行原子化提取并去重
- `getRelationsGroup`：对源数据的边进行原子化提取并去重
- `getNodesInfo`：打印节点相关信息

    ```
    python -m scripts.getNodesGroup
    python -m scripts.getNodesInfo
    python -m scripts.getRelationsGroup
    ```

- `verify_node_text_match`：检查处理后的原子节点与文本的匹配程度

    ```
    # 该脚本只能用于大致检测，无法精确分析，其结果仅供参考
    python -m scripts.verify_node_text_match
    ```

- `test_chunk`：测试文本分块效果
    ```
    python -m scripts.test_chunk
    ```

- `build_sft_datasets`：构建 phase：0 节点训练 SFT 数据集
    ```
    # 在 config/data.yaml 中设置好参数
    python -m scripts.build_sft_dataset
    ```