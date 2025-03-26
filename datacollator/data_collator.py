import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class DSSMDataCollator:
    """
    handle input from both user and item parts, generalize input_ids
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = "longest"
    max_user_length: int = 1536
    max_item_length: int = 192
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        user_features = {
            "input_ids": [f["user_input_ids"] for f in features],
            "attention_mask": [f["user_attention_mask"] for f in features]
        }

        item_features = {
            "input_ids": [f["item_input_ids"] for f in features],
            "attention_mask": [f["item_attention_mask"] for f in features]
        }
        # user_texts = [f["user_text"] for f in features]
        # item_texts = [f["item_text"] for f in features]

        user_batch = self._pad_features(
            user_features,
            max_length=self.max_user_length,
        )
        
        item_batch = self._pad_features(
            item_features,
            max_length=self.max_item_length,
        )

        batch = {
            "user_input_ids": user_batch["input_ids"],
            "user_attention_mask": user_batch["attention_mask"],
            "item_input_ids": item_batch["input_ids"],
            "item_attention_mask": item_batch["attention_mask"],
        }
        
        if "labels" in features[0]:
            batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
            
        return batch
    
    def _pad_features(self, features, max_length):
        """辅助方法：对特定侧的特征进行填充"""
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        return batch