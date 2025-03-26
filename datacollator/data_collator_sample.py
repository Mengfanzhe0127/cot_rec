import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class DSSMDataCollatorSample:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = "longest"
    max_user_length: int = 1536
    max_item_length: int = 192
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    num_negatives: int = 127

    def __call__(self, features: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        
        # ==================== 用户侧处理 ====================
        user_input_ids = [f["user_input_ids"] for f in features]
        user_attention_mask = [f["user_attention_mask"] for f in features]
        
        user_batch = self.tokenizer.pad(
            {
                "input_ids": user_input_ids,
                "attention_mask": user_attention_mask
            },
            padding=self.padding,
            max_length=self.max_user_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        item_input_ids = torch.stack([torch.tensor(f["item_input_ids"]) for f in features])
        item_attention_mask = torch.stack([torch.tensor(f["item_attention_mask"]) for f in features])

        return {
            "user_input_ids": user_batch["input_ids"],
            "user_attention_mask": user_batch["attention_mask"],
            "item_input_ids": item_input_ids,
            "item_attention_mask": item_attention_mask,
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
        }
        
        ###############################################################################################################
        # ==================== 原物品侧处理 ====================
        # item_input_ids = torch.cat([torch.tensor(f["item_input_ids"]) for f in features])        # 三维结构：[B][K+1][L]
        # item_attention_mask = torch.cat([torch.tensor(f["item_attention_mask"]) for f in features])

        # # 全局填充（确保同一批次内长度一致）
        # padded_item = self.tokenizer.pad(
        #     {
        #         "input_ids": item_input_ids,
        #         "attention_mask": item_attention_mask
        #     },
        #     padding=self.padding,
        #     max_length=self.max_item_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors=self.return_tensors,
        # )

        # # 重构三维结构 [B, K+1, L]
        # batch_size = len(features)
        # num_items = 1 + self.num_negatives
        # return {
        #     "user_input_ids": user_batch["input_ids"],
        #     "user_attention_mask": user_batch["attention_mask"],
        #     "item_input_ids": padded_item["input_ids"].view(batch_size, num_items, -1),
        #     "item_attention_mask": padded_item["attention_mask"].view(batch_size, num_items, -1),
        #     "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
        # }
        ###############################################################################################################
    
@dataclass
class OptimizedDataCollator:
    tokenizer: PreTrainedTokenizerBase
    item_embeddings: torch.Tensor
    num_negatives: int = 127
    max_user_length: int = 1536
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # 处理用户侧
        user_batch = self.tokenizer.pad(
            {
                "input_ids": [f["user_input_ids"] for f in features],
                "attention_mask": [f["user_attention_mask"] for f in features]
            },
            padding='longest',
            max_length=self.max_user_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        all_indices = torch.arange(len(self.item_embeddings))
        neg_indices = []
        
        for f in features:
            pos_idx = f["positive_idx"]
            mask = torch.ones_like(all_indices, dtype=torch.bool)
            mask[pos_idx] = False
            candidates = all_indices[mask]
            neg = candidates[torch.randperm(len(candidates))[:self.num_negatives]]
            neg_indices.append(neg)
        
        # 收集嵌入
        pos_emb = self.item_embeddings[[f["positive_idx"] for f in features]]
        neg_emb = self.item_embeddings[torch.stack(neg_indices)]
        
        return {
            "user_input_ids": user_batch["input_ids"],
            "user_attention_mask": user_batch["attention_mask"],
            "positive_embeddings": pos_emb,
            "negative_embeddings": neg_emb,
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long)
        }