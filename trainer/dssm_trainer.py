from transformers import Trainer
from typing import Optional, List, Dict, Union, Tuple, Any
from datasets import Dataset
import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)

from utils.utils import compute_all_item_embeddings

class DSSMTrainer(Trainer):
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
            # 获取用户的embedding
            user_inputs = {
                'input_ids': inputs['user_input_ids'].to(self.args.device),
                'attention_mask': inputs['user_attention_mask'].to(self.args.device)
            }
            
            user_embeddings = model.encode_user(
                input_ids=user_inputs['input_ids'],
                attention_mask=user_inputs['attention_mask']
            )
            
            # 获取全局电影embedding并计算相似度
            item_embeddings = self.item_embedding_matrix.to(user_embeddings.device)
            
            # 计算用户与所有电影的相似度
            similarity = torch.matmul(user_embeddings, item_embeddings.t()) / self.similarity_temperature
            
            # 获取全局标签，用于评估
            labels = inputs.get('labels').to(self.args.device)

            loss = None
            if labels is not None and not prediction_loss_only:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(similarity, labels)
            
        return (loss, similarity, labels)