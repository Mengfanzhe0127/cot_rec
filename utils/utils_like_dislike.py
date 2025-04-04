from tqdm import tqdm
import torch
import logging

logger = logging.getLogger(__name__)

def create_movie_text(movie_name, movie_info_dict):
    """create structured movie text from movie name and movie info dict"""
    movie_name = movie_name.strip()
    assert movie_name in movie_info_dict, f"Movie name {movie_name} not found in movie info dict"

    info = movie_info_dict[movie_name]
    movie_text = f"movie: {movie_name}\n"
    # movie_text += f"title: {info['title']}\n"
    movie_text += f"year: {info['year']}\n"
    movie_text += f"genre: {', '.join(info['genre'])}\n"
    movie_text += f"director: {', '.join(info['director'])}\n"
    movie_text += f"writer: {', '.join(info['writer'])}\n"
    movie_text += f"star: {', '.join(info['star'])}\n"
    # movie_text += f"plot: {info['plot']}\n"

    return movie_text

def add_instruction_prompt(texts, instruction, is_query=True):
    """
    为文本添加指令提示
    texts: 文本列表
    instruction: 指令内容
    is_query: 是否为查询文本
    """
    if is_query:
        return [f"Instruct: {instruction}\nQuery: {text}" if text.strip() else "" for text in texts]
    return texts

def compute_all_item_embeddings(model, tokenizer, movie_list, movie_info_dict, batch_size=32, max_length=192, device="cuda"):
    """Calculate all movie embeddings"""
    movie_texts = [create_movie_text(movie, movie_info_dict) for movie in movie_list]
    
    embeddings_list = []
    for i in tqdm(range(0, len(movie_texts), batch_size), desc="Computing item embeddings"):
        batch_texts = movie_texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            padding='longest',  # 使用一致的填充策略
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            batch_embeddings = model.encode_item(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        
        embeddings_list.append(batch_embeddings)
    
    return torch.cat(embeddings_list, dim=0)