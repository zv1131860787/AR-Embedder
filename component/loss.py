from typing import Dict

import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature: float, weight: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.weight = weight

    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = torch.matmul(query_embeddings, doc_embeddings.transpose(0, 1)) / self.temperature
        target = torch.arange(logits.size(0), device=logits.device)
        loss_q2d = F.cross_entropy(logits, target)
        loss_d2q = F.cross_entropy(logits.transpose(0, 1), target)
        contrastive_loss = 0.5 * (loss_q2d + loss_d2q)
        total = self.weight * contrastive_loss
        return {
            "loss": total,
            "contrastive_loss": contrastive_loss.detach(),
        }
