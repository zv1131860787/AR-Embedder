import math
from typing import Dict, Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput


class BertLayerForDecoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_shape)

        attention_output = self.dense(context_layer)
        attention_output = self.output_dropout(attention_output)
        attention_output = self.attention_layer_norm(attention_output + query)

        intermediate_output = self.intermediate_act_fn(self.intermediate(attention_output))
        layer_output = self.output_dense(intermediate_output)
        layer_output = self.output_dropout2(layer_output)
        layer_output = self.output_layer_norm(layer_output + attention_output)
        return layer_output


class RetroMAEModel(nn.Module):
    def __init__(self, model_name_or_path: str, projection_dim: Optional[int] = None) -> None:
        super().__init__()
        self.lm = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        config = self.lm.config
        if not hasattr(self.lm, "bert"):
            raise ValueError("RetroMAEModel currently supports BERT-like masked LM backbones.")
        self.decoder_embeddings = self.lm.bert.embeddings
        self.c_head = BertLayerForDecoder(config)
        self.c_head.apply(self.lm._init_weights)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.projection = None
        if projection_dim and projection_dim != config.hidden_size:
            self.projection = nn.Linear(config.hidden_size, projection_dim)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        encoder_labels: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        decoder_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        lm_out: MaskedLMOutput = self.lm(
            encoder_input_ids,
            attention_mask=encoder_attention_mask,
            labels=encoder_labels,
            output_hidden_states=True,
            return_dict=True,
        )
        cls_hiddens = lm_out.hidden_states[-1][:, :1, :]
        decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids)
        hiddens = torch.cat([cls_hiddens, decoder_embedding_output[:, 1:, :]], dim=1)

        decoder_position_ids = self.lm.bert.embeddings.position_ids[:, : decoder_input_ids.size(1)]
        decoder_position_embeddings = self.lm.bert.embeddings.position_embeddings(decoder_position_ids)
        query = decoder_position_embeddings + cls_hiddens

        matrix_attention_mask = self.lm.get_extended_attention_mask(
            decoder_attention_mask,
            decoder_attention_mask.shape,
            decoder_attention_mask.device,
        )

        decoder_hidden = self.c_head(
            query=query,
            key=hiddens,
            value=hiddens,
            attention_mask=matrix_attention_mask,
        )

        pred_scores = self.lm.cls(decoder_hidden)
        decoder_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            decoder_labels.view(-1),
        )
        total_loss = decoder_loss + lm_out.loss

        outputs = {
            "loss": total_loss,
            "encoder_mlm_loss": lm_out.loss.detach(),
            "decoder_mlm_loss": decoder_loss.detach(),
        }
        if self.projection is not None:
            outputs["cls_projection"] = self.projection(cls_hiddens.squeeze(1))
        return outputs

    def save_pretrained(self, output_dir: str) -> None:
        self.lm.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, projection_dim: Optional[int] = None) -> "RetroMAEModel":
        return cls(model_name_or_path=model_name_or_path, projection_dim=projection_dim)


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


class CustomRetroMAEModel(RetroMAEModel):
    def __init__(
        self,
        model_name_or_path: str,
        projection_dim: Optional[int] = None,
        doc_encoder_name_or_path: Optional[str] = None,
    ) -> None:
        super().__init__(model_name_or_path=model_name_or_path, projection_dim=projection_dim)
        if doc_encoder_name_or_path:
            doc_model = AutoModelForMaskedLM.from_pretrained(doc_encoder_name_or_path)
            if not hasattr(doc_model, "bert"):
                raise ValueError("Custom document encoder must be BERT-like to provide embeddings.")
            if doc_model.config.vocab_size != self.lm.config.vocab_size:
                raise ValueError("Custom document encoder must share vocabulary size with the encoder backbone.")
            self.decoder_embeddings = doc_model.bert.embeddings
            if hasattr(self.lm, "cls") and hasattr(self.lm.cls, "predictions"):
                self.lm.cls.predictions.decoder.weight = self.decoder_embeddings.word_embeddings.weight
