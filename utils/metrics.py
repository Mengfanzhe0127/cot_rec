import numpy as np
from typing import Dict, Tuple

def calculate_recall(predictions: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    Recall@k (Hit@k)
    
    Args:
        predictions: shape=[batch_size, num_items]
        labels: shape=[batch_size]
        k: top k recommendations
        
    Returns:
        Recall@k
    """
    # 获取Top-K预测
    top_k_preds = predictions[:, :k]
    
    # 判断每个样本的真实标签是否在Top-K预测中
    hits = np.any(top_k_preds == labels.reshape(-1, 1), axis=1).astype(int)
    
    return np.mean(hits)

def calculate_mrr(predictions: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    MRR@k
    
    Args:
        predictions: shape=[batch_size, num_items]
        labels: shape=[batch_size]
        k: top k recommendations
        
    Returns:
        MRR@k
    """
    top_k_preds = predictions[:, :k]
    hits = (top_k_preds == labels.reshape(-1, 1))

    # 计算排名：找到第一个命中的索引（从1开始）
    ranks = np.argmax(hits, axis=1) + 1  # +1 是因为排名从1开始

    # 无命中的设为0（MRR 计算时未命中的应为 0）
    ranks[~np.any(hits, axis=1)] = 0  

    # 计算 Reciprocal Rank：1/排名，对于无命中样本保持为 0
    mrr_scores = np.zeros_like(ranks, dtype=float)
    valid_mask = ranks > 0
    mrr_scores[valid_mask] = 1.0 / ranks[valid_mask]

    return np.mean(mrr_scores)

def calculate_ndcg(predictions: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    NDCG@k
    
    Args:
        predictions: shape=[batch_size, num_items]
        labels: shape=[batch_size]
        k: top k recommendations
        
    Returns:
        NDCG@k
    """
    # 获取Top-K预测
    top_k_preds = predictions[:, :k]
    hits = (top_k_preds == labels.reshape(-1, 1))

    ranks = np.argmax(hits, axis=1).astype(float) + 1
    ranks[~np.any(hits, axis=1)] = np.inf  # 无命中的设为 inf

    # 计算DCG
    dcg = np.zeros_like(ranks, dtype=float)
    mask = ranks != np.inf
    dcg[mask] = 1.0 / np.log2(ranks[mask] + 1)

    # 计算IDCG（最佳情况）
    idcg = 1.0 / np.log2(2)  # 只有一个相关项，IDCG始终是 1.0
    
    return np.mean(dcg / idcg)

def compute_recommendation_metrics(
    predictions: np.ndarray, 
    labels: np.ndarray, 
    k_values: Tuple[int] = (1, 5, 10, 20, 50)
) -> Dict[str, float]:
    """
    计算推荐系统的评估指标
    
    Args:
        predictions: 预测分数矩阵 [样本数, 类别数]
        labels: 真实标签 [样本数]
        k_values: 要计算的k值列表
        
    Returns:
        包含各项指标的字典
    """
    assert predictions.shape[0] == labels.shape[0]

    metrics = {}

    # 获取预测的Top-K索引（按预测分数从高到低排序）
    top_indices = np.argsort(-predictions, axis=1)

    # 为每个k值计算指标
    for k in k_values:
        # 计算Recall@k (Hit@k)
        metrics[f"recall@{k}"] = calculate_recall(top_indices, labels, k)
        metrics[f"mrr@{k}"] = calculate_mrr(top_indices, labels, k)
        metrics[f"ndcg@{k}"] = calculate_ndcg(top_indices, labels, k)
    
    return metrics