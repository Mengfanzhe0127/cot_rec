from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

@dataclass
class DSSMDataCollatorPreference:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = "longest"
    max_user_length: int = 1536
    max_item_length: int = 192
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    num_negatives: int = 127

    def __call__(self, features: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        
        # ==================== Preference文本处理 ====================
        preference_input_ids = [f["preference_input_ids"] for f in features]
        preference_attention_mask = [f["preference_attention_mask"] for f in features]
        
        preference_batch = self.tokenizer.pad(
            {
                "input_ids": preference_input_ids,
                "attention_mask": preference_attention_mask
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
            "preference_input_ids": preference_batch["input_ids"],
            "preference_attention_mask": preference_batch["attention_mask"],
            "item_input_ids": item_input_ids,
            "item_attention_mask": item_attention_mask,
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
        }