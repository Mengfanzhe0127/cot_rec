import torch
import torch.nn as nn
import torch.nn.functional as F
from .dssm_model_base import DSSMBase

class Qwen2DSSMLikeDislike(DSSMBase):
    """
    The two-tower model with like-dislike mechanism without projection layers
    """

    def forward(
        self,
        like_input_ids=None,
        like_attention_mask=None,
        dislike_input_ids=None,
        dislike_attention_mask=None,
        item_input_ids=None,
        item_attention_mask=None,
        labels=None,
    ):
        batch_size = like_input_ids.size(0)
        items_per_sample = item_input_ids.size(1)

        # 判断输入是否为空字符串
        like_is_empty = (like_attention_mask.sum(dim=1) <= 1)
        dislike_is_empty = (dislike_attention_mask.sum(dim=1) <= 1)

        like_embeddings = self.encode_user(
            input_ids=like_input_ids,
            attention_mask=like_attention_mask,
        ) # [batch_size, hidden_size]
        
        dislike_embeddings = self.encode_user(
            input_ids=dislike_input_ids,
            attention_mask=dislike_attention_mask,
        ) # [batch_size, hidden_size]
            
        item_shape = item_input_ids.shape
        item_embeddings = self.encode_item(
            input_ids=item_input_ids.view(-1, item_shape[-1]),
            attention_mask=item_attention_mask.view(-1, item_shape[-1])
        ).view(batch_size, items_per_sample, -1)  # [B, 1+K, hidden_size]

        like_similarity = self.compute_similarity(like_embeddings, item_embeddings)  # [B, 1+K]

        like_similarity = like_similarity * (~like_is_empty).unsqueeze(1).float()

        dislike_similarity = self.compute_similarity(dislike_embeddings, item_embeddings)  # [B, 1+K]

        dislike_similarity = dislike_similarity * (~dislike_is_empty).unsqueeze(1).float()
        
        # final similarity = like_similarity - dislike_similarity
        similarity = like_similarity - dislike_similarity

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # positive sample is at the first position
            targets = torch.zeros(batch_size, dtype=torch.long, device=similarity.device)
            loss = loss_fct(similarity, targets)

        return {
            "loss": loss,
            "similarity": similarity,
            "like_embeddings": like_embeddings,
            "dislike_embeddings": dislike_embeddings, 
            "item_embeddings": item_embeddings,
        }