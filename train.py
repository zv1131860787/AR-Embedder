import logging
import random

import numpy as np
import torch
from transformers import AutoTokenizer

from component.argument import parse_args
from component.dataset import (
    ContrastiveCollator,
    ContrastiveDataset,
    RetroARCollator,
    RetroARDataset,
    RetroMAECollator,
    RetroMAEDataset,
)
from component.model import ContrastiveEncoderModel, CustomRetroMAEModel, RetroARModel, RetroMAEModel
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
    elif args.training_task == "retroar":
        dataset = RetroARDataset(
            file_path=args.train_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            mask_ratio=args.doc_mask_ratio,
        )
        collator = RetroARCollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=8 if args.fp16 else None,
        )
        model = RetroARModel(
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
    elif args.training_task == "contrastive":
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
    else:
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
        model = CustomRetroMAEModel(
            model_name_or_path=args.model_name_or_path,
            projection_dim=args.projection_dim,
            doc_encoder_name_or_path=args.custom_doc_encoder,
        )
        _resize_token_embeddings(
            model,
            tokenizer,
            tie_lm_head=True,
        )
        trainer = RetroMAETrainer(
            model=model,
            args=args,
            dataset=dataset,
            collate_fn=collator,
            tokenizer=tokenizer,
        )
    trainer.train()


def _resize_token_embeddings(model, tokenizer, tie_lm_head: bool, resize_doc_encoder: bool = True) -> None:
    if hasattr(model, "lm"):
        model.lm.resize_token_embeddings(len(tokenizer))
        return
    if not hasattr(model, "query_encoder"):
        return
    if hasattr(model.query_encoder, "resize_token_embeddings"):
        query_embedding = model.query_encoder.get_input_embeddings()
        if query_embedding is not None and query_embedding.weight.size(0) != len(tokenizer):
            model.query_encoder.resize_token_embeddings(len(tokenizer))
    if resize_doc_encoder and hasattr(model, "doc_encoder"):
        doc_embeddings = None
        if hasattr(model.doc_encoder, "get_input_embeddings"):
            doc_embeddings = model.doc_encoder.get_input_embeddings()
        if hasattr(model.doc_encoder, "resize_token_embeddings"):
            if doc_embeddings is None or doc_embeddings.weight.size(0) != len(tokenizer):
                model.doc_encoder.resize_token_embeddings(len(tokenizer))
    if tie_lm_head and hasattr(model, "lm_head") and hasattr(model, "doc_encoder"):
        doc_embeddings = model.doc_encoder.get_input_embeddings()
        if doc_embeddings is not None:
            model.lm_head.weight = doc_embeddings.weight


if __name__ == "__main__":
    main()
