import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class RetroMAEDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_seq_length: int,
        mask_ratio: float,
    ) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Training file not found: {self.file_path}")
        if mask_ratio <= 0 or mask_ratio >= 1:
            raise ValueError("mask_ratio must be within (0, 1)")
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mask_ratio = mask_ratio
        self._samples: List[Dict[str, List[int]]] = []
        self._load()

    def _load(self) -> None:
        with self.file_path.open("r", encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                query = record.get("query")
                document = record.get("document") or record.get("doc") or record.get("passage")
                if not query:
                    raise ValueError("Each training instance must provide a `query` field.")
                if not document:
                    document = query
                encoded_query = self.tokenizer(
                    query,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_attention_mask=True,
                    return_token_type_ids=True,
                )
                encoded_doc = self.tokenizer(
                    document,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_attention_mask=True,
                    return_token_type_ids=True,
                )
                masked_doc = self._mask_tokens(encoded_doc["input_ids"])
                self._samples.append(
                    {
                        "query_input_ids": encoded_query["input_ids"],
                        "query_attention_mask": encoded_query["attention_mask"],
                        "query_token_type_ids": encoded_query.get("token_type_ids"),
                        "doc_input_ids": masked_doc["input_ids"],
                        "doc_attention_mask": encoded_doc["attention_mask"],
                        "doc_token_type_ids": encoded_doc.get("token_type_ids"),
                        "doc_labels": masked_doc["labels"],
                    }
                )

    def _mask_tokens(self, input_ids: List[int]) -> Dict[str, List[int]]:
        special_ids = set(self.tokenizer.all_special_ids)
        cand_indexes = [i for i, token_id in enumerate(input_ids) if token_id not in special_ids]
        if not cand_indexes:
            raise ValueError("Input example does not contain maskable tokens.")
        num_to_mask = max(1, math.floor(len(cand_indexes) * self.mask_ratio))
        mask_indices = set(random.sample(cand_indexes, num_to_mask))
        masked_input_ids = list(input_ids)
        labels = [-100] * len(input_ids)
        for idx in mask_indices:
            original_token = input_ids[idx]
            labels[idx] = original_token
            prob = random.random()
            if prob < 0.8:
                masked_input_ids[idx] = self.tokenizer.mask_token_id
            elif prob < 0.9:
                masked_input_ids[idx] = random.randint(0, self.tokenizer.vocab_size - 1)
            else:
                masked_input_ids[idx] = original_token
        return {"input_ids": masked_input_ids, "labels": labels}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self._samples[index]


class RetroMAECollator:
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch = {}
        batch["query_input_ids"] = self._pad(features, "query_input_ids", self.tokenizer.pad_token_id)
        batch["query_attention_mask"] = self._pad(features, "query_attention_mask", 0)
        if features[0].get("query_token_type_ids") is not None:
            batch["query_token_type_ids"] = self._pad(features, "query_token_type_ids", 0)
        batch["doc_input_ids"] = self._pad(features, "doc_input_ids", self.tokenizer.pad_token_id)
        batch["doc_attention_mask"] = self._pad(features, "doc_attention_mask", 0)
        if features[0].get("doc_token_type_ids") is not None:
            batch["doc_token_type_ids"] = self._pad(features, "doc_token_type_ids", 0)
        batch["doc_labels"] = self._pad(features, "doc_labels", -100)
        return batch

    def _pad(self, features: List[Dict[str, List[int]]], key: str, pad_value: int) -> torch.Tensor:
        values = [feature[key] for feature in features]
        max_len = max(len(item) for item in values)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            if max_len % multiple != 0:
                max_len = (max_len // multiple + 1) * multiple
        padded = [item + [pad_value] * (max_len - len(item)) for item in values]
        return torch.tensor(padded, dtype=torch.long)


class ContrastiveDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_seq_length: int) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Training file not found: {self.file_path}")
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self._samples: List[Dict[str, List[int]]] = []
        self._load()

    def _load(self) -> None:
        with self.file_path.open("r", encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                query = record.get("query")
                document = record.get("document") or record.get("doc") or record.get("passage")
                if not query:
                    raise ValueError("Each contrastive instance must provide a `query` field.")
                if not document:
                    document = query
                encoded_query = self.tokenizer(
                    query,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_attention_mask=True,
                    return_token_type_ids=True,
                )
                encoded_doc = self.tokenizer(
                    document,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_attention_mask=True,
                    return_token_type_ids=True,
                )
                self._samples.append(
                    {
                        "query_input_ids": encoded_query["input_ids"],
                        "query_attention_mask": encoded_query["attention_mask"],
                        "query_token_type_ids": encoded_query.get("token_type_ids"),
                        "doc_input_ids": encoded_doc["input_ids"],
                        "doc_attention_mask": encoded_doc["attention_mask"],
                        "doc_token_type_ids": encoded_doc.get("token_type_ids"),
                    }
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self._samples[index]


class ContrastiveCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {}
        batch["query_input_ids"] = self._pad(features, "query_input_ids", self.tokenizer.pad_token_id)
        batch["query_attention_mask"] = self._pad(features, "query_attention_mask", 0)
        if features[0].get("query_token_type_ids") is not None:
            batch["query_token_type_ids"] = self._pad(features, "query_token_type_ids", 0)
        batch["doc_input_ids"] = self._pad(features, "doc_input_ids", self.tokenizer.pad_token_id)
        batch["doc_attention_mask"] = self._pad(features, "doc_attention_mask", 0)
        if features[0].get("doc_token_type_ids") is not None:
            batch["doc_token_type_ids"] = self._pad(features, "doc_token_type_ids", 0)
        return batch

    def _pad(self, features: List[Dict[str, List[int]]], key: str, pad_value: int) -> torch.Tensor:
        values = [feature[key] for feature in features]
        max_len = max(len(item) for item in values)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            if max_len % multiple != 0:
                max_len = (max_len // multiple + 1) * multiple
        padded = [item + [pad_value] * (max_len - len(item)) for item in values]
        return torch.tensor(padded, dtype=torch.long)
