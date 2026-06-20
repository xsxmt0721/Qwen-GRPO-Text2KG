### 快速部署
- 下载 flash-attn 预编译包
    ```
	wget --content-disposition -nv \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    ```
- 构造镜像/容器
    ```
    BUILDKIT=1 docker compose build --progress=plain
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

- `build_sft_datasets`：构建 phase：0 节点训练 SFT 数据集和 GRPO 的数据集
    ```
    # 在 config/data.yaml 中设置好参数
    python -m scripts.build_sft_dataset
    ```

- `split_sft_dataset`：对 sft/grpo 数据集进行划分
    ```
    # 在 config/data.yaml 中设置好参数
    python -m scripts.split_sft_dataset -d sft
    # 或
    python -m scripts.split_sft_dataset -d grpo
    ```

- `completion_length_stats`：统计数据集中 completion 的平均长度（token数）
    ```
    python -m scripts.completion_length_stats Output/SFTDatasets/sft_node_train.json --model /models/Qwen2.5-1.5B-Instruct
    ```

- `sft_text2kg`：进行 sft 监督微调
    ```
    # 在 config/sft.yaml 中设置好参数
    python -m scripts.sft_text2kg
    ```

- `text2kg_eval`：对 sft 微调 / grpo 训练的结果进行测试（默认为 sft 微调结果）
    ```
    # 全量测试
    python -m scripts.text2kg_eval
    # 快速测试：指定测试集尺寸
    python -m scripts.text2kg_eval -n 100
    # 快速测试 + 贪心解码
    python -m scripts.text2kg_eval -n 50 --disable-sampling
    # 测试 grpo 训练结果需要指定 config，否则默认测试 sft 数据集
    python -m scripts.text2kg_eval --config config/grpo_nodes.yaml -n 100
    # 如果需要测试指定 checkpoint，加入 --checkpoint，否则固定选择最优模型
    python -m scripts.text2kg_eval --config config/grpo_nodes.yaml --checkpoint 45 -n 100      
    ```

- `test_node_reward`: 测试节点奖励函数能否正常跑通，默认跑 SFTDatasets
    ```
    # 对照测试: 验证奖励函数自身正确性
    # -n：指定数量
    # --control：对照控制组，加入该项后将会启动 label 内部比较，用于自测 reward 函数的正确性和标签本身的质量
    # --detail：控制是否输出逐条详情
    python scripts/test_node_reward.py --control -n 20 --detail
    # 跑 GRPO 测试结果的示例如下：
    python scripts/test_node_reward.py \
  --train-file Output/GRPODatasets/grpo_node_train.json \
  --raw-file Output/GRPODatasets/grpo_node_test_raw.json
    ```

- `grpo_text2kg`：运行 grpo，仅从文本中提取节点
    ```
    # 全量训练
    python scripts/grpo_text2kg.py

    # 快速测试（仅 100 条 + 30 条验证集）
    python scripts/grpo_text2kg.py -n 200 --max-eval-samples 30

    # 指定配置
    python scripts/grpo_text2kg.py --config config/grpo_nodes.yaml
    ```