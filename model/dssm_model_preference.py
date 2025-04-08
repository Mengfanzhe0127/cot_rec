import torch
import torch.nn as nn
import torch.nn.functional as F
from .dssm_model_base import DSSMBase

class Qwen2DSSMPreference(DSSMBase):
    """
    The two-tower model using single preference field for matching
    """
    def __init__(self, config):
        super().__init__(config)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        preference_input_ids=None,
        preference_attention_mask=None,
        item_input_ids=None,
        item_attention_mask=None,
        labels=None,
    ):
        batch_size = preference_input_ids.size(0)
        items_per_sample = item_input_ids.size(1)

        preference_is_empty = (preference_attention_mask.sum(dim=1) <= 1)

        preference_embeddings = self.encode_user(
            input_ids=preference_input_ids,
            attention_mask=preference_attention_mask,
        ) # [batch_size, hidden_size]
            
        item_shape = item_input_ids.shape
        item_embeddings = self.encode_item(
            input_ids=item_input_ids.view(-1, item_shape[-1]),
            attention_mask=item_attention_mask.view(-1, item_shape[-1])
        ).view(batch_size, items_per_sample, -1)  # [B, 1+K, hidden_size]

        preference_similarity = self.compute_similarity(preference_embeddings, item_embeddings)  # [B, 1+K]

        # preference="", set similarity to 0.5
        similarity = torch.where(
            preference_is_empty.unsqueeze(1),
            torch.ones_like(preference_similarity) * 0.5,
            preference_similarity
        )

        loss = None
        if labels is not None:
            targets = torch.zeros(batch_size, dtype=torch.long, device=similarity.device)
            loss = self.ce_loss(similarity, targets)

        return {
            "loss": loss,
            "similarity": similarity,
            "preference_embeddings": preference_embeddings,
            "item_embeddings": item_embeddings,
            "preference_is_empty": preference_is_empty
        }