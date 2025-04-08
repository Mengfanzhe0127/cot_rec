from transformers import Trainer
from typing import Optional, List, Dict, Union, Tuple, Any
from datasets import Dataset
import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)

from utils.utils_like_dislike import compute_all_item_embeddings

class DSSMTrainerPreference(Trainer):
    def __init__(self, similarity_temperature = 0.07, movie_list = None, movie_info_dict = None, data_args = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.item_embedding_matrix = None  # 初始化为None，在评估和预测时计算全部的item embedding
        self.movie_list = movie_list
        self.movie_to_idx = {movie: idx for idx, movie in enumerate(movie_list)}
        # eval / predict使用全部的item embedding
        self.movie_info_dict = movie_info_dict
        self.data_args = data_args
        self.similarity_temperature = similarity_temperature
        self.logger = logger
        
        # 只定义交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss()

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            ):
        self.logger.info("Eval: Recalculating item embeddings for evaluation with current model weights...")

        self.item_embedding_matrix = compute_all_item_embeddings(
            model=self.model,
            tokenizer=self.tokenizer,
            movie_list=self.movie_list,
            movie_info_dict=self.movie_info_dict,
            batch_size=self.data_args.item_batch_size,
            max_length=self.data_args.item_max_length,
            device=self.args.device
        )

        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
    
    def predict(
            self,
            test_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "test",
            ):
        self.logger.info("Predict: Recalculating item embeddings for prediction with current model weights...")

        self.item_embedding_matrix = compute_all_item_embeddings(
            model=self.model,
            tokenizer=self.tokenizer,
            movie_list=self.movie_list,
            movie_info_dict=self.movie_info_dict,
            batch_size=self.data_args.item_batch_size,
            max_length=self.data_args.item_max_length,
            device=self.args.device
        )

        return super().predict(
            test_dataset=test_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        预测：全局电影
        """
        self.model.eval()
        with torch.no_grad():
            # 获取preference的embedding
            preference_is_empty = (inputs['preference_attention_mask'].sum(dim=1) <= 1).to(self.args.device)

            preference_inputs = {
                'input_ids': inputs['preference_input_ids'].to(self.args.device),
                'attention_mask': inputs['preference_attention_mask'].to(self.args.device)
            }
            
            preference_embeddings = model.encode_user(**preference_inputs)
            
            # 获取全局电影embedding
            item_embeddings = self.item_embedding_matrix.to(preference_embeddings.device)
            
            # 计算相似度
            preference_similarity = torch.matmul(preference_embeddings, item_embeddings.t()) / self.similarity_temperature

            # 对于preference为空的样本，将相似度设为0.5 (中立状态)
            similarity = torch.where(
                preference_is_empty.unsqueeze(1),
                torch.ones_like(preference_similarity) * 0.5,  # 如果为空，则设为0.5
                preference_similarity  # 如果不为空，则保持原值
            )
            
            # 获取全局标签，用于评估
            labels = inputs.get('labels').to(self.args.device)

            loss = None
            if labels is not None and not prediction_loss_only:
                # 简化: 对所有样本使用交叉熵损失
                loss = self.ce_loss(similarity, labels)
            
        return (loss, similarity, labels)