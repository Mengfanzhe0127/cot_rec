import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel, Qwen2Model

class Qwen2DSSMLikeDislike(Qwen2PreTrainedModel):
    """
    The two-tower model with like-dislike mechanism without projection layers
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.similarity_temperature = config.similarity_temperature
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
        
    def _get_pooled_output(self, hidden_states, input_ids, attention_mask):
        """获取序列的池化表示，使用最后一个非padding token的隐藏状态"""
        batch_size = input_ids.shape[0]
        
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("If a padding token is not defined, you cannot handle batch_size > 1")
            
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]  # 防止没有padding token的情况
                sequence_lengths = sequence_lengths.to(hidden_states.device)
        
        pooled_output = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        return pooled_output
    
    def encode_text(self, input_ids, attention_mask, position_ids=None, inputs_embeds=None, use_cache=False,
                   output_attentions=None, output_hidden_states=None, return_dict=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        pooled_output = self._get_pooled_output(hidden_states, input_ids, attention_mask)

        embeddings = F.normalize(pooled_output, p=2, dim=-1)
        return embeddings
    
    def encode_user(self, input_ids, attention_mask, **kwargs):
        """
        处理like或dislike空文本
        """
        is_empty = (input_ids.sum(dim=1) == 0)

        if is_empty.any():
            batch_size = input_ids.shape[0]
            empty_embeddings = torch.zeros(
                batch_size, 
                self.config.hidden_size, 
                device=input_ids.device, 
                dtype=next(self.parameters()).dtype
            )
        
            # 如果全部为空，直接返回
            if is_empty.all():
                return empty_embeddings
            
            non_empty_indices = torch.where(~is_empty)[0]
            non_empty_input_ids = input_ids[non_empty_indices]
            non_empty_attention_mask = attention_mask[non_empty_indices]

            non_empty_embeddings = self.encode_text(non_empty_input_ids, non_empty_attention_mask, **kwargs)

            result_embeddings = empty_embeddings.clone()
            result_embeddings[non_empty_indices] = non_empty_embeddings
            return result_embeddings

        # 正常情况下的处理
        return self.encode_text(input_ids, attention_mask, **kwargs)
    
    def encode_item(self, input_ids, attention_mask, **kwargs):
        return self.encode_text(input_ids, attention_mask, **kwargs)
    
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

        like_is_empty = (like_input_ids.sum(dim=1) == 0)
        dislike_is_empty = (dislike_input_ids.sum(dim=1) == 0)

        # 编码like和dislike文本
        like_embeddings = self.encode_user(
            input_ids=like_input_ids,
            attention_mask=like_attention_mask,
        ) # [batch_size, hidden_size]
        
        dislike_embeddings = self.encode_user(
            input_ids=dislike_input_ids,
            attention_mask=dislike_attention_mask,
        ) # [batch_size, hidden_size]
            
        # 编码item文本
        item_shape = item_input_ids.shape
        item_embeddings = self.encode_item(
            input_ids=item_input_ids.view(-1, item_shape[-1]),
            attention_mask=item_attention_mask.view(-1, item_shape[-1])
        ).view(batch_size, items_per_sample, -1)  # [B, 1+K, hidden_size]

        # 计算like与item的相似度
        like_similarity = torch.einsum(
            'bd,bkd->bk', 
            like_embeddings, 
            item_embeddings
        ) / self.similarity_temperature  # [B, 1+K]

        # 对于like为空的样本，将相似度设为0
        like_similarity = like_similarity * (~like_is_empty).unsqueeze(1).float()
        
        # 计算dislike与item的相似度
        dislike_similarity = torch.einsum(
            'bd,bkd->bk', 
            dislike_embeddings, 
            item_embeddings
        ) / self.similarity_temperature  # [B, 1+K]

        # 对于dislike为空的样本，将相似度设为0
        dislike_similarity = dislike_similarity * (~dislike_is_empty).unsqueeze(1).float()
        
        # 计算最终相似度: like相似度 - dislike相似度
        similarity = like_similarity - dislike_similarity

        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 正例在第一个位置
            targets = torch.zeros(batch_size, dtype=torch.long, device=similarity.device)
            loss = loss_fct(similarity, targets)

        return {
            "loss": loss,
            "similarity": similarity,
            "like_embeddings": like_embeddings,
            "dislike_embeddings": dislike_embeddings, 
            "item_embeddings": item_embeddings,
        }