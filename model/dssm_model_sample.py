import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel, Qwen2Model

class Qwen2DSSMSample(Qwen2PreTrainedModel):
    """
    The two-tower model, based on Qwen2Model, adds user and commodity towers for the recommendation system
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)

        self.dnn_output_dim = config.dnn_output_dim
        self.similarity_temperature = config.similarity_temperature
        
        self.user_projection = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, self.dnn_output_dim)
        )

        self.item_projection = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, self.dnn_output_dim)
        )

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
                   output_attentions=None, output_hidden_states=None, return_dict=None, projection=None):
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
        
        if projection is not None:
            embeddings = projection(pooled_output)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings
        return pooled_output
    
    def encode_user(self, input_ids, attention_mask, **kwargs):
        return self.encode_text(input_ids, attention_mask, projection=self.user_projection, **kwargs)
    
    def encode_item(self, input_ids, attention_mask, **kwargs):
        return self.encode_text(input_ids, attention_mask, projection=self.item_projection, **kwargs)
    
    def forward(
        self,
        user_input_ids=None,
        user_attention_mask=None,
        item_input_ids=None,
        item_attention_mask=None,
        labels=None,
    ):
        batch_size = user_input_ids.size(0)
        items_per_sample = item_input_ids.size(1)
        
        user_embeddings = self.encode_user(
            input_ids=user_input_ids,
            attention_mask=user_attention_mask,
        ) # [batch_size, hidden_size]
            
        item_shape = item_input_ids.shape
        item_embeddings = self.encode_item(
            input_ids=item_input_ids.view(-1, item_shape[-1]),  # [batch_size*(1 + K), sequence_length] -> [batch_size*(1+K)*sequence_length] 
            attention_mask=item_attention_mask.view(-1, item_shape[-1])
        ).view(batch_size, items_per_sample, -1)  # [B, 1+K, hidden_size]

        similarity = torch.einsum(
            'bd,bkd->bk', 
            user_embeddings, 
            item_embeddings
        ) / self.similarity_temperature  # [B, 1+K]

        # 4. 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 正例在第一个位置
            targets = torch.zeros(batch_size, dtype=torch.long, device=similarity.device)
            loss = loss_fct(similarity, targets)

        return {
            "loss": loss,
            "similarity": similarity,
            "user_embeddings": user_embeddings,
            "item_embeddings": item_embeddings,
        }
                
                