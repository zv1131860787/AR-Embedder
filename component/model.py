from typing import Dict, Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class RetroMAEModel(nn.Module):
    def __init__(self, model_name_or_path: str, projection_dim: Optional[int] = None) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_name_or_path)
        self.query_encoder = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.doc_encoder = AutoModel.from_pretrained(model_name_or_path, config=config)
        hidden_size = config.hidden_size
        projection_dim = projection_dim or hidden_size
        self.query_projection = nn.Linear(hidden_size, projection_dim)
        self.doc_projection = nn.Linear(hidden_size, projection_dim)
        self.lm_head = nn.Linear(hidden_size, config.vocab_size, bias=False)
        self.lm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.register_parameter("bias", self.lm_bias)
        self.lm_head.weight = self.doc_encoder.get_input_embeddings().weight

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        query_token_type_ids: Optional[torch.Tensor] = None,
        doc_token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        query_outputs = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            token_type_ids=query_token_type_ids,
            return_dict=True,
        )
        doc_outputs = self.doc_encoder(
            input_ids=doc_input_ids,
            attention_mask=doc_attention_mask,
            token_type_ids=doc_token_type_ids,
            return_dict=True,
        )
        query_cls = query_outputs.last_hidden_state[:, 0, :]
        doc_cls = doc_outputs.last_hidden_state[:, 0, :]
        query_embeddings = nn.functional.normalize(self.query_projection(query_cls), dim=-1)
        doc_embeddings = nn.functional.normalize(self.doc_projection(doc_cls), dim=-1)
        doc_logits = self.lm_head(doc_outputs.last_hidden_state) + self.bias
        return {
            "query_embeddings": query_embeddings,
            "doc_embeddings": doc_embeddings,
            "doc_logits": doc_logits,
        }


class ContrastiveEncoderModel(nn.Module):
    def __init__(self, model_name_or_path: str, projection_dim: Optional[int] = None) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_name_or_path)
        self.query_encoder = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.doc_encoder = AutoModel.from_pretrained(model_name_or_path, config=config)
        hidden_size = config.hidden_size
        projection_dim = projection_dim or hidden_size
        self.query_projection = nn.Linear(hidden_size, projection_dim)
        self.doc_projection = nn.Linear(hidden_size, projection_dim)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        query_token_type_ids: Optional[torch.Tensor] = None,
        doc_token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        query_outputs = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            token_type_ids=query_token_type_ids,
            return_dict=True,
        )
        doc_outputs = self.doc_encoder(
            input_ids=doc_input_ids,
            attention_mask=doc_attention_mask,
            token_type_ids=doc_token_type_ids,
            return_dict=True,
        )
        query_cls = query_outputs.last_hidden_state[:, 0, :]
        doc_cls = doc_outputs.last_hidden_state[:, 0, :]
        query_embeddings = nn.functional.normalize(self.query_projection(query_cls), dim=-1)
        doc_embeddings = nn.functional.normalize(self.doc_projection(doc_cls), dim=-1)
        return {
            "query_embeddings": query_embeddings,
            "doc_embeddings": doc_embeddings,
        }
