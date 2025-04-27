from transformers import Trainer
from typing import Optional, List, Dict, Union, Tuple, Any
from datasets import Dataset
import torch
from torch import nn
import logging
import numpy as np
import json
from pathlib import Path

logger = logging.getLogger(__name__)

from utils.utils_like_dislike import compute_all_item_embeddings

class DSSMTrainerPreference(Trainer):
    def __init__(self, similarity_temperature = 0.07, movie_list = None, movie_info_dict = None, data_args = None, *args, **kwargs):
        # self.steps_to_log = kwargs.pop("steps_to_log", None) # 用于记录梯度爆炸对应的步数
        super().__init__(*args, **kwargs)
        self.item_embedding_matrix = None
        self.movie_list = movie_list
        self.movie_to_idx = {movie.strip(): idx for idx, movie in enumerate(movie_list)}
        # eval / predict使用全部的item embedding
        self.movie_info_dict = movie_info_dict
        self.data_args = data_args
        self.similarity_temperature = similarity_temperature
        self.logger = logger

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

        eval_results = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

        if hasattr(self, "eval_metrics") and hasattr(self.eval_metrics, "predictions"):
            predictions = self.eval_metrics.predictions
            labels = self.eval_metrics.label_ids
            self.save_prediction_results(predictions, labels, eval_dataset, prefix="eval")

        return eval_results
    
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

        predict_output = super().predict(
            test_dataset=test_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

        predictions = predict_output.predictions
        labels = predict_output.label_ids
        self.save_prediction_results(predictions, labels, test_dataset, prefix="test")
    
        return predict_output
        
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
    
    def save_prediction_results(self, predictions, labels, dataset, prefix = "eval"):
        """预测结果保存到jsonl"""
        k_values = (1, 5, 10, 20, 50)
        top_indices = np.argsort(-predictions, axis=1)

        results = []
        for i, (pred, label) in enumerate(zip(predictions, labels)):
            sample = dataset[i]
        
            preference = ""
            if "preference_input_ids" in sample:
                preference_ids = sample["preference_input_ids"]
                preference = self.tokenizer.decode(preference_ids, skip_special_tokens=True)
                # 移除前缀指令，只保留实际偏好内容
                if "Given a user's movie preferences" in preference:
                    try:
                        preference = preference.split("Query: ")[1].strip()
                    except IndexError:
                        # 如果分割失败，保留完整文本
                        pass

            # 获取目标电影名称 - 从标签索引获取
            if 0 <= label < len(self.movie_list):
                target_movie = self.movie_list[label]
            else:
                target_movie = f"未知电影(索引:{label})"

            top_50_indices = top_indices[i, :50]
            predicted_movies = [self.movie_list[idx] for idx in top_50_indices if idx < len(self.movie_list)]

            success_1 = int(label in top_indices[i, :1])
            success_5 = int(label in top_indices[i, :5])
            success_10 = int(label in top_indices[i, :10])
            success_20 = int(label in top_indices[i, :20])
            success_50 = int(label in top_indices[i, :50])
        
            # 计算各项指标
            recall_values = []
            mrr_values = []
            ndcg_values = []

            for k in k_values:
                # Recall@k
                hits_k = np.any(top_indices[i, :k] == label)
                recall_values.append(float(hits_k))

                # MRR@k
                if hits_k:
                    rank = np.where(top_indices[i, :k] == label)[0][0] + 1
                    mrr_values.append(1.0 / rank)
                else:
                    mrr_values.append(0.0)
            
                # NDCG@k
                if hits_k:
                    rank = np.where(top_indices[i, :k] == label)[0][0] + 1
                    dcg = 1.0 / np.log2(rank + 1)
                    idcg = 1.0 / np.log2(2)
                    ndcg_values.append(float(dcg / idcg))
                else:
                    ndcg_values.append(0.0)
        
            # 构建结果 - 不再包含id字段
            result = {
                "preference": preference,
                "success_1": success_1,
                "success_5": success_5,
                "success_10": success_10,
                "success_20": success_20,
                "success_50": success_50,
                "recall": recall_values,
                "mrr": mrr_values,
                "ndcg": ndcg_values,
                "predict": predicted_movies,
                "target_movie": target_movie
            }

            results.append(result)

        # 保存结果
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"{prefix}_predictions.jsonl"

        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
        print(f"Saved {len(results)} prediction results to {output_file}")