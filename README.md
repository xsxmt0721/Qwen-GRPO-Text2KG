### 快速部署
- 构造镜像/容器
    ```
    BUILDKIT=1 docker compose build
    docker compose up -d
    ```
- 进入容器
    ```
    docker exec -it kgrl /bin/bash
    ```

### 基座模型下载

- 创建模型存储总目录
    ```
    cd ~
    mkdir KGRL_Models
    ```
- 创建基座模型的目录
    ```
    cd KGRL_Models
    mkdir qwen2.5-1.5B-Instruct
    ```
- 下载模型到本地：
    ```
    hf download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir ~/KGRL_Models/Qwen2.5-1.5B-Instruct
    ```

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

- `split_sft_dataset`：对 sft 数据集进行划分
    ```
    # 在 config/data.yaml 中设置好参数
    python -m scripts.split_sft_dataset
    ```

- `sft_text2kg`：进行 sft 监督微调
    ```
    # 在 config/sft.yaml 中设置好参数
    python -m scripts.sft_text2kg
    ```

- `sft_text2kg_eval`：对 sft 微调的结果进行测试
    ```
    python -m scripts.sft_text2kg_eval
    ```