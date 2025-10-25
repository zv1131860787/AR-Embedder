import logging
import math
import os
from typing import Dict

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .loss import ContrastiveLoss

logger = logging.getLogger(__name__)


class RetroMAETrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        args,
        dataset,
        collate_fn,
        tokenizer: AutoTokenizer,
    ) -> None:
        self.model = model
        self.args = args
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn = None

    def train(self) -> None:
        os.makedirs(self.args.output_dir, exist_ok=True)
        train_dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
        total_update_steps = self._compute_train_steps(len(train_dataloader))
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_eps,
            weight_decay=self.args.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_update_steps * self.args.warmup_ratio),
            num_training_steps=total_update_steps,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.fp16)
        self.model.train()
        self._log_start(len(train_dataloader), total_update_steps)
        global_step = 0
        optimizer.zero_grad()
        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    outputs = self.model(
                        encoder_input_ids=batch["encoder_input_ids"],
                        encoder_attention_mask=batch["encoder_attention_mask"],
                        encoder_labels=batch["encoder_labels"],
                        decoder_input_ids=batch["decoder_input_ids"],
                        decoder_attention_mask=batch["decoder_attention_mask"],
                        decoder_labels=batch["decoder_labels"],
                    )
                    log_losses = {
                        "loss": outputs["loss"].detach(),
                        "encoder_mlm_loss": outputs["encoder_mlm_loss"],
                        "decoder_mlm_loss": outputs["decoder_mlm_loss"],
                    }
                    loss = outputs["loss"] / self.args.gradient_accumulation_steps
                scaler.scale(loss).backward()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if global_step % self.args.logging_steps == 0:
                        self._log_step(global_step, log_losses)
                    if global_step % self.args.save_steps == 0:
                        self._save_checkpoint(global_step)
                    if global_step >= total_update_steps:
                        break
            if global_step >= total_update_steps:
                break
        self._save_checkpoint(global_step, final=True)

    def _compute_train_steps(self, dataloader_len: int) -> int:
        steps_per_epoch = math.ceil(dataloader_len / self.args.gradient_accumulation_steps)
        return steps_per_epoch * self.args.num_train_epochs

    def _log_start(self, dataloader_len: int, total_steps: int) -> None:
        logger.info("Starting training")
        logger.info("Number of samples: %s", len(self.dataset))
        logger.info("Steps per epoch: %s", dataloader_len)
        logger.info("Total optimization steps: %s", total_steps)

    def _log_step(self, step: int, losses: Dict[str, torch.Tensor]) -> None:
        logger.info(
            "step=%s loss=%.4f encoder_mlm=%.4f decoder_mlm=%.4f",
            step,
            losses["loss"].item(),
            losses["encoder_mlm_loss"].item(),
            losses["decoder_mlm_loss"].item(),
        )

    def _save_checkpoint(self, step: int, final: bool = False) -> None:
        tag = "final" if final else f"step-{step}"
        output_dir = os.path.join(self.args.output_dir, tag)
        os.makedirs(output_dir, exist_ok=True)
        state = {
            "model_state_dict": self.model.state_dict(),
            "tokenizer": self.tokenizer.name_or_path,
            "step": step,
        }
        torch.save(state, os.path.join(output_dir, "checkpoint.pt"))


class ContrastiveTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        args,
        dataset,
        collate_fn,
        tokenizer: AutoTokenizer,
    ) -> None:
        self.model = model
        self.args = args
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn = ContrastiveLoss(
            temperature=args.temperature,
            weight=args.contrastive_weight,
        )
        self.loss_fn.to(self.device)

    def train(self) -> None:
        os.makedirs(self.args.output_dir, exist_ok=True)
        train_dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
        total_update_steps = self._compute_train_steps(len(train_dataloader))
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_eps,
            weight_decay=self.args.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_update_steps * self.args.warmup_ratio),
            num_training_steps=total_update_steps,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.fp16)
        self.model.train()
        self._log_start(len(train_dataloader), total_update_steps)
        global_step = 0
        optimizer.zero_grad()
        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    outputs = self.model(
                        query_input_ids=batch["query_input_ids"],
                        query_attention_mask=batch["query_attention_mask"],
                        doc_input_ids=batch["doc_input_ids"],
                        doc_attention_mask=batch["doc_attention_mask"],
                        query_token_type_ids=batch.get("query_token_type_ids"),
                        doc_token_type_ids=batch.get("doc_token_type_ids"),
                    )
                    losses = self.loss_fn(
                        outputs["query_embeddings"],
                        outputs["doc_embeddings"],
                    )
                    loss = losses["loss"] / self.args.gradient_accumulation_steps
                scaler.scale(loss).backward()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if global_step % self.args.logging_steps == 0:
                        self._log_step(global_step, losses)
                    if global_step % self.args.save_steps == 0:
                        self._save_checkpoint(global_step)
                    if global_step >= total_update_steps:
                        break
            if global_step >= total_update_steps:
                break
        self._save_checkpoint(global_step, final=True)

    def _compute_train_steps(self, dataloader_len: int) -> int:
        steps_per_epoch = math.ceil(dataloader_len / self.args.gradient_accumulation_steps)
        return steps_per_epoch * self.args.num_train_epochs

    def _log_start(self, dataloader_len: int, total_steps: int) -> None:
        logger.info("Starting contrastive training")
        logger.info("Number of samples: %s", len(self.dataset))
        logger.info("Steps per epoch: %s", dataloader_len)
        logger.info("Total optimization steps: %s", total_steps)

    def _log_step(self, step: int, losses: Dict[str, torch.Tensor]) -> None:
        logger.info(
            "step=%s loss=%.4f contrastive=%.4f",
            step,
            losses["loss"].item(),
            losses["contrastive_loss"].item(),
        )

    def _save_checkpoint(self, step: int, final: bool = False) -> None:
        tag = "final" if final else f"step-{step}"
        output_dir = os.path.join(self.args.output_dir, tag)
        os.makedirs(output_dir, exist_ok=True)
        state = {
            "model_state_dict": self.model.state_dict(),
            "tokenizer": self.tokenizer.name_or_path,
            "step": step,
        }
        torch.save(state, os.path.join(output_dir, "checkpoint.pt"))
