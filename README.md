# AR-Embedder

An embedding pre-training playground that implements several retrieval-oriented objectives:

- **RetroMAE** – masked auto-encoder style dual-encoder pre-training.
- **RetroAR** – encoder–decoder variant with an autoregressive decoder and CLS-conditioned prefix.
- **Contrastive** – dual-encoder bidirectional contrastive learning.
- **Custom RetroMAE** – RetroMAE with a pluggable document encoder.

The repository is structured to make it easy to switch between these tasks without changing the
surrounding training pipeline.

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended) or CPU

Install runtime dependencies:

```bash
pip install torch transformers datasets tensorboardX tqdm
```

> `tensorboardX` is optional, but required if you want TensorBoard logging (`--log_dir`).

## Project Layout

```
AR-Embedder/
├── component/
│   ├── argument.py         # CLI / dataclass definitions
│   ├── dataset.py          # RetroMAE / RetroAR / contrastive datasets & collators
│   ├── model.py            # RetroMAE, RetroAR, Custom RetroMAE, contrastive models
│   ├── trainer.py          # Trainers with AMP, schedulers, tensorboard logging
│   ├── loss.py             # Contrastive loss (shared by contrastive trainer)
├── dataset/
│   └── train_sample.jsonl  # Example training data (query-document pairs)
├── train.py                # Entry point
└── README.md
```

## Data Preparation

Datasets are expected in **JSON Lines** format. Each line must contain at least a `query` field and
one of `document`, `doc`, or `passage`.

`dataset/train_sample.jsonl` demonstrates the expected format:

```jsonl
{"query": "如何制作红烧肉？", "document": "红烧肉通常先焯水，再用生抽老抽冰糖慢炖，最后大火收汁即可。"}
{"query": "RetroMAE 的核心目标是什么", "document": "RetroMAE 通过掩码自编码器的重建和对比学习，使得查询和文档的表示保持一致。"}
{"query": "北京有什么著名景点", "document": "北京的故宫、长城、颐和园和天坛都是很受欢迎的景点。"}
```

## Usage

All training modes share the same CLI. Choose the objective with `--training_task`.

### RetroMAE Pre-training

```bash
python train.py \
  --training_task retromae \
  --model_name_or_path bert-base-chinese \
  --train_file dataset/train_sample.jsonl \
  --output_dir checkpoints/retromae
```

RetroMAE applies masked language modeling on the encoder and decoder branches and optimises a
contrastive objective.

### RetroAR (Autoregressive Decoder)

```bash
python train.py \
  --training_task retroar \
  --model_name_or_path bert-base-chinese \
  --train_file dataset/train_sample.jsonl \
  --output_dir checkpoints/retroar
```

RetroAR masks the encoder input but trains the decoder autoregressively. The CLS embedding from the
encoder is prepended to the decoder input as context, and its label is automatically set to `-100`.

### Contrastive Dual-Encoder

```bash
python train.py \
  --training_task contrastive \
  --model_name_or_path bert-base-chinese \
  --train_file dataset/train_sample.jsonl \
  --output_dir checkpoints/contrastive
```

Only contrastive alignment between query and document embeddings is optimised in this mode.

### Custom RetroMAE

Use a separate document encoder (must be a BERT-like masked-LM with the same vocabulary size):

```bash
python train.py \
  --training_task custom \
  --model_name_or_path bert-base-chinese \
  --custom_doc_encoder your-doc-encoder \
  --train_file dataset/train_sample.jsonl \
  --output_dir checkpoints/custom
```

## Key Arguments

| Argument | Description |
| --- | --- |
| `--model_name_or_path` | Hugging Face model ID or path for the query encoder (and doc encoder, unless `--custom_doc_encoder`). |
| `--train_file` | JSONL training file with `query` and `document` fields. |
| `--output_dir` | Directory for checkpoints (`step-*/checkpoint.pt`) and final model. |
| `--training_task` | `retromae`, `retroar`, `contrastive`, or `custom`. |
| `--max_seq_length` | Max sequence length for tokenization. |
| `--doc_mask_ratio` | Mask ratio for encoder/decoder inputs (RetroMAE & RetroAR encoders). |
| `--projection_dim` | Optional projection layer size for embeddings. |
| `--per_device_train_batch_size` | DataLoader batch size per device. |
| `--gradient_accumulation_steps` | Effective batch multiplier via gradient accumulation. |
| `--learning_rate` | Base learning rate for AdamW. |
| `--log_dir` | If set, enables TensorBoardX logging at the given path. |
| `--fp16` | Enable AMP mixed-precision training. |

Run `python train.py --help` for the complete set of options.

## Logging & Monitoring

- Training progress is logged via Python logging (per `--logging_steps`).
- When `--log_dir` is specified, metrics are written with TensorBoardX (`train/loss`, `train/encoder_mlm_loss`,
  `train/decoder` or `train/contrastive_loss`). Visualise with:

  ```bash
  tensorboard --logdir <log_dir>
  ```

## Checkpoints

Checkpoints are saved to `--output_dir` every `--save_steps` steps and once at the end. Each checkpoint folder contains:

- `checkpoint.pt` – `state_dict`, tokenizer name, and global step.

## Tips

- RetroAR expects an autoregressive decoder – ensure your dataset does not already contain `<mask>` tokens in the document field.
- `datasets` streaming is not used; the JSONL file is loaded into memory. For large corpora, consider memory size or adapt `_load` to stream.
- Mixed precision (`--fp16`) can dramatically speed up training on CUDA devices.

## License

This repository does not currently specify a license. Add one if you intend to share or publish
derivative work.

---

# AR-Embedder（中文说明）

AR-Embedder 是一个用于构建检索型表示（embedding）的预训练项目，支持以下训练策略：

- **RetroMAE**：参照 RetroMAE 论文的 masked auto-encoder 方案，同时优化查询与文档向量的对比学习目标。
- **RetroAR**：在 RetroMAE 基础上替换为自回归解码器，利用 CLS 向量作为解码前缀，解码端按自回归方式训练。
- **Contrastive**：纯粹的双塔对比学习，仅优化查询-文档的相似度。
- **Custom RetroMAE**：自定义文档编码器（必须为与主模型同词表的 BERT 类 masked LM）。

## 环境依赖

- Python 3.9 及以上
- 推荐使用支持 CUDA 的 GPU

安装依赖：

```bash
pip install torch transformers datasets tensorboardX tqdm
```

> 如需写 TensorBoard 日志，请确保安装了 `tensorboardX`。

## 目录结构

```
AR-Embedder/
├── component/
│   ├── argument.py         # 命令行参数/数据类
│   ├── dataset.py          # RetroMAE / RetroAR / Contrastive 的数据集与 Collator
│   ├── model.py            # RetroMAE、RetroAR、Custom RetroMAE、Contrastive 模型
│   ├── trainer.py          # 训练器（含 AMP、调度器、TensorBoardX）
│   ├── loss.py             # 对比损失
├── dataset/
│   └── train_sample.jsonl  # 数据示例
├── train.py                # 训练入口
└── README.md
```

## 数据格式

训练数据使用 JSON Lines，每行至少包含 `query` 字段和 `document`（或 `doc`/`passage`）字段。示例：

```jsonl
{"query": "如何制作红烧肉？", "document": "红烧肉通常先焯水，再用生抽老抽冰糖慢炖，最后大火收汁即可。"}
{"query": "RetroMAE 的核心目标是什么", "document": "RetroMAE 通过掩码自编码器的重建和对比学习，使得查询和文档的表示保持一致。"}
{"query": "北京有什么著名景点", "document": "北京的故宫、长城、颐和园和天坛都是很受欢迎的景点。"}
```

## 训练示例

### RetroMAE

```bash
python train.py \
  --training_task retromae \
  --model_name_or_path bert-base-chinese \
  --train_file dataset/train_sample.jsonl \
  --output_dir checkpoints/retromae
```

### RetroAR

```bash
python train.py \
  --training_task retroar \
  --model_name_or_path bert-base-chinese \
  --train_file dataset/train_sample.jsonl \
  --output_dir checkpoints/retroar
```

该模式中，encoder 输入仍做随机掩码；decoder 采用自回归训练，CLS 向量仅用于提供上下文，不计算损失。

### 对比学习

```bash
python train.py \
  --training_task contrastive \
  --model_name_or_path bert-base-chinese \
  --train_file dataset/train_sample.jsonl \
  --output_dir checkpoints/contrastive
```

### 自定义文档编码器

```bash
python train.py \
  --training_task custom \
  --model_name_or_path bert-base-chinese \
  --custom_doc_encoder your-doc-encoder \
  --train_file dataset/train_sample.jsonl \
  --output_dir checkpoints/custom
```

## 常用参数

| 参数 | 说明 |
| --- | --- |
| `--model_name_or_path` | 预训练模型的名称或路径（查询编码器）。 |
| `--train_file` | JSONL 训练数据路径。 |
| `--output_dir` | 输出目录，用于保存 checkpoint。 |
| `--training_task` | `retromae` / `retroar` / `contrastive` / `custom`。 |
| `--max_seq_length` | Tokenizer 的最大长度。 |
| `--doc_mask_ratio` | 文档/查询的掩码比例（RetroMAE / RetroAR 的 encoder 部分）。 |
| `--projection_dim` | 嵌入投影维度，可选。 |
| `--per_device_train_batch_size` | 单设备 batch 大小。 |
| `--gradient_accumulation_steps` | 梯度累积步数。 |
| `--learning_rate` | AdamW 学习率。 |
| `--log_dir` | 指定目录即开启 TensorBoard 日志。 |
| `--fp16` | 启用 AMP 混合精度。 |

使用 `python train.py --help` 可查看全部参数。

## 日志与监控

- 控制台每隔 `--logging_steps` 输出一次指标。
- 指定 `--log_dir` 时，会使用 TensorBoardX 写入标量，可通过 `tensorboard --logdir <log_dir>` 查看。

## Checkpoint

模型按 `--save_steps` 定期保存，最终会额外保存一次。每个 checkpoint 目录包含 `checkpoint.pt`，里面包含模型参数、tokenizer 名称与训练步数。

## 注意事项

- RetroAR 的 decoder 采用 teacher forcing，自回归的标签与输入需要严格对齐。数据中不应提前包含 `[MASK]` 等特殊标记。
- 当前数据加载会将 JSONL 全部读取到内存；如需处理超大数据集，可改造 `_load` 函数使用流式读取。
- 使用 GPU 并启用 `--fp16` 通常能显著加速训练。

## 许可证

仓库暂未指定开源许可证。如果需要发布或共享，请根据需求补充许可证信息。
