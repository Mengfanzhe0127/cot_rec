from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

@dataclass
class DSSMDataCollatorLikeDislike:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = "longest"
    max_user_length: int = 1536
    max_item_length: int = 192
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    num_negatives: int = 127

    def __call__(self, features: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        
        # ==================== Like文本处理 ====================
        like_input_ids = [f["like_input_ids"] for f in features]
        like_attention_mask = [f["like_attention_mask"] for f in features]
        
        like_batch = self.tokenizer.pad(
            {
                "input_ids": like_input_ids,
                "attention_mask": like_attention_mask
            },
            padding=self.padding,
            max_length=self.max_user_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # ==================== Dislike文本处理 ====================
        dislike_input_ids = [f["dislike_input_ids"] for f in features]
        dislike_attention_mask = [f["dislike_attention_mask"] for f in features]
        
        dislike_batch = self.tokenizer.pad(
            {
                "input_ids": dislike_input_ids,
                "attention_mask": dislike_attention_mask
            },
            padding=self.padding,
            max_length=self.max_user_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # ==================== 物品文本处理 ====================
        item_input_ids = torch.stack([torch.tensor(f["item_input_ids"]) for f in features])
        item_attention_mask = torch.stack([torch.tensor(f["item_attention_mask"]) for f in features])

        return {
            "like_input_ids": like_batch["input_ids"],
            "like_attention_mask": like_batch["attention_mask"],
            "dislike_input_ids": dislike_batch["input_ids"],
            "dislike_attention_mask": dislike_batch["attention_mask"],
            "item_input_ids": item_input_ids,
            "item_attention_mask": item_attention_mask,
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
        }