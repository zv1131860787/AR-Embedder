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
