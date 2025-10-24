from typing import Dict

import torch
import torch.nn.functional as F


class RetroMAELoss(torch.nn.Module):
    def __init__(self, temperature: float, contrastive_weight: float, mlm_weight: float) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.mlm_weight = mlm_weight

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        doc_logits: torch.Tensor,
        doc_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits = torch.matmul(query_embeddings, doc_embeddings.transpose(0, 1)) / self.temperature
        target = torch.arange(logits.size(0), device=logits.device)
        loss_q2d = F.cross_entropy(logits, target)
        loss_d2q = F.cross_entropy(logits.transpose(0, 1), target)
        contrastive_loss = 0.5 * (loss_q2d + loss_d2q)
        vocab_size = doc_logits.size(-1)
        mlm_loss = F.cross_entropy(
            doc_logits.view(-1, vocab_size),
            doc_labels.view(-1),
            ignore_index=-100,
        )
        total = self.contrastive_weight * contrastive_loss + self.mlm_weight * mlm_loss
        return {
            "loss": total,
            "contrastive_loss": contrastive_loss.detach(),
            "mlm_loss": mlm_loss.detach(),
        }
