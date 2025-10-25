import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainArguments:
    model_name_or_path: str
    tokenizer_name: Optional[str]
    train_file: str
    output_dir: str
    max_seq_length: int
    training_task: str
    doc_mask_ratio: float
    projection_dim: int
    temperature: float
    mlm_weight: float
    contrastive_weight: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    max_grad_norm: float
    num_train_epochs: int
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    seed: int
    fp16: bool
    dataloader_num_workers: int
    custom_doc_encoder: Optional[str]


def parse_args() -> TrainArguments:
    parser = argparse.ArgumentParser(description="Embedding training entrypoint.")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-chinese")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument(
        "--training_task",
        type=str,
        default="retromae",
        choices=["retromae", "contrastive", "custom"],
        help="Choose RetroMAE pretraining or dual-encoder contrastive training.",
    )
    parser.add_argument("--doc_mask_ratio", type=float, default=0.3)
    parser.add_argument("--projection_dim", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--mlm_weight", type=float, default=1.0)
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.98)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument(
        "--custom_doc_encoder",
        type=str,
        default=None,
        help="Optional Hugging Face model name/path used as the document encoder for custom training.",
    )
    args = parser.parse_args()
    return TrainArguments(
        model_name_or_path=args.model_name_or_path,
        tokenizer_name=args.tokenizer_name,
        train_file=args.train_file,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        training_task=args.training_task,
        doc_mask_ratio=args.doc_mask_ratio,
        projection_dim=args.projection_dim,
        temperature=args.temperature,
        mlm_weight=args.mlm_weight,
        contrastive_weight=args.contrastive_weight,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        custom_doc_encoder=args.custom_doc_encoder,
    )
