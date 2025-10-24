import logging
import random

import numpy as np
import torch
from transformers import AutoTokenizer

from component.argument import parse_args
from component.dataset import (
    ContrastiveCollator,
    ContrastiveDataset,
    RetroMAECollator,
    RetroMAEDataset,
)
from component.model import ContrastiveEncoderModel, RetroMAEModel
from component.trainer import ContrastiveTrainer, RetroMAETrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()
    set_seed(args.seed)
    tokenizer_name = args.tokenizer_name or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if args.training_task == "retromae":
        dataset = RetroMAEDataset(
            file_path=args.train_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            mask_ratio=args.doc_mask_ratio,
        )
        collator = RetroMAECollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=8 if args.fp16 else None,
        )
        model = RetroMAEModel(
            model_name_or_path=args.model_name_or_path,
            projection_dim=args.projection_dim,
        )
        _resize_token_embeddings(model, tokenizer, tie_lm_head=True)
        trainer = RetroMAETrainer(
            model=model,
            args=args,
            dataset=dataset,
            collate_fn=collator,
            tokenizer=tokenizer,
        )
    else:
        dataset = ContrastiveDataset(
            file_path=args.train_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
        collator = ContrastiveCollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=8 if args.fp16 else None,
        )
        model = ContrastiveEncoderModel(
            model_name_or_path=args.model_name_or_path,
            projection_dim=args.projection_dim,
        )
        _resize_token_embeddings(model, tokenizer, tie_lm_head=False)
        trainer = ContrastiveTrainer(
            model=model,
            args=args,
            dataset=dataset,
            collate_fn=collator,
            tokenizer=tokenizer,
        )
    trainer.train()


def _resize_token_embeddings(model, tokenizer, tie_lm_head: bool) -> None:
    if not hasattr(model, "query_encoder"):
        return
    query_embedding = model.query_encoder.get_input_embeddings()
    if query_embedding.weight.size(0) == len(tokenizer):
        return
    model.query_encoder.resize_token_embeddings(len(tokenizer))
    if hasattr(model, "doc_encoder"):
        model.doc_encoder.resize_token_embeddings(len(tokenizer))
    if tie_lm_head and hasattr(model, "lm_head"):
        model.lm_head.weight = model.doc_encoder.get_input_embeddings().weight


if __name__ == "__main__":
    main()
