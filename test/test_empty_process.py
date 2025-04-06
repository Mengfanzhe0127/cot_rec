import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
import numpy as np

# 模拟数据
sample_data = [
    {"like": "movies: Avengers (2012), Iron Man (2008)\nattributes: action, superhero", "dislike": ""},
    {"like": "", "dislike": "movies: The Room (2003)\nattributes: bad acting, poor plot"},
    {"like": "movies: Star Wars (1977)\nattributes: sci-fi", "dislike": "movies: Twilight (2008)"},
    {"like": "", "dislike": ""}
]

def test_empty_string_handling():
    print("====== 测试空字符串处理 ======")
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("/mnt/wangxiaolei/model/Qwen/gte-Qwen2-7B-instruct", trust_remote_code=True)
    
    # 设置参数
    max_seq_length = 128  # 根据您的实际设置调整
    
    # 模拟预处理函数处理
    print("\n1. 模拟预处理函数处理:")
    
    # 提取like和dislike文本
    like_texts = [item["like"] for item in sample_data]
    dislike_texts = [item["dislike"] for item in sample_data]
    
    # 添加提示模板
    like_texts_with_prompt = [
        f"Given a user's movie preferences, retrieve relevant movies that match these preferences\nQuery: {text}" if text.strip() else ""
        for text in like_texts
    ]
    
    dislike_texts_with_prompt = [
        f"Given a user's movie preferences, retrieve relevant movies that match these preferences\nQuery: {text}" if text.strip() else ""
        for text in dislike_texts
    ]
    
    # 打印原始文本和提示模板后的文本
    for i in range(len(sample_data)):
        print(f"\n样本 {i+1}:")
        print(f"  like原文: '{like_texts[i]}'")
        print(f"  dislike原文: '{dislike_texts[i]}'")
        print(f"  like添加提示后: '{like_texts_with_prompt[i]}'")
        print(f"  dislike添加提示后: '{dislike_texts_with_prompt[i]}'")
        print(f"  like是否为空: {not bool(like_texts[i].strip())}")
        print(f"  dislike是否为空: {not bool(dislike_texts[i].strip())}")
    
    # 进行分词处理，使用与您程序一致的padding策略
    like_encodings = tokenizer(
        like_texts_with_prompt,
        padding="longest",  # 使用最长序列的长度进行填充
        max_length=max_seq_length,
        truncation=True,
        return_tensors="pt"
    )
    
    dislike_encodings = tokenizer(
        dislike_texts_with_prompt,
        padding="longest",
        max_length=max_seq_length,
        truncation=True,
        return_tensors="pt"
    )
    
    # 打印编码后的like和dislike输入
    print("\n2. 分词处理结果:")
    print(f"like_input_ids形状: {like_encodings['input_ids'].shape}")
    print(f"dislike_input_ids形状: {dislike_encodings['input_ids'].shape}")
    
    # 分析每个样本的输入
    for i in range(len(sample_data)):
        print(f"\n样本 {i+1} 的编码:")
        # 提取当前样本的编码
        like_ids = like_encodings['input_ids'][i]
        like_mask = like_encodings['attention_mask'][i]
        dislike_ids = dislike_encodings['input_ids'][i]
        dislike_mask = dislike_encodings['attention_mask'][i]
        
        # 统计非填充token数量
        like_non_pad_count = like_mask.sum().item()
        dislike_non_pad_count = dislike_mask.sum().item()
        
        # 找出非填充token的位置和ID
        like_non_pad_positions = torch.where(like_mask == 1)[0].tolist()
        dislike_non_pad_positions = torch.where(dislike_mask == 1)[0].tolist()
        
        like_non_pad_tokens = like_ids[like_non_pad_positions].tolist()
        dislike_non_pad_tokens = dislike_ids[dislike_non_pad_positions].tolist()
        
        # 计算input_ids总和
        like_sum = like_ids.sum().item()
        dislike_sum = dislike_ids.sum().item()
        
        print(f"  like_input_ids: {like_ids[:10].tolist()}...")
        print(f"  dislike_input_ids: {dislike_ids[:10].tolist()}...")
        print(f"  like_input_ids总和: {like_sum}")
        print(f"  dislike_input_ids总和: {dislike_sum}")
        print(f"  like非填充token数量: {like_non_pad_count}")
        print(f"  dislike非填充token数量: {dislike_non_pad_count}")
        
        # 详细显示非填充token信息
        print(f"  like非填充token位置: {like_non_pad_positions}")
        print(f"  like非填充token ID: {like_non_pad_tokens}")
        
        print(f"  dislike非填充token位置: {dislike_non_pad_positions}")
        print(f"  dislike非填充token ID: {dislike_non_pad_tokens}")
        
        # 尝试解码非填充token
        try:
            like_tokens_decoded = tokenizer.decode(like_non_pad_tokens)
            dislike_tokens_decoded = tokenizer.decode(dislike_non_pad_tokens)
            print(f"  like非填充token解码: '{like_tokens_decoded}'")
            print(f"  dislike非填充token解码: '{dislike_tokens_decoded}'")
        except:
            print("  无法解码token")
            
        # 特别关注空输入的情况
        if not like_texts[i].strip():  # like为空
            print("  【特别注意】like为空字符串，对应的非填充token ID: ", like_non_pad_tokens)
            # 查看填充token
            pad_id = tokenizer.pad_token_id
            print(f"  填充token ID: {pad_id}")
            # 解码特殊token
            for token_id in like_non_pad_tokens:
                try:
                    token_str = tokenizer.decode([token_id])
                    print(f"    Token ID {token_id} => '{token_str}'")
                except:
                    print(f"    Token ID {token_id} 无法解码")
            
        if not dislike_texts[i].strip():  # dislike为空
            print("  【特别注意】dislike为空字符串，对应的非填充token ID: ", dislike_non_pad_tokens)
            # 查看特殊token含义
            for token_id in dislike_non_pad_tokens:
                try:
                    token_str = tokenizer.decode([token_id])
                    print(f"    Token ID {token_id} => '{token_str}'")
                except:
                    print(f"    Token ID {token_id} 无法解码")
    
    # 打印tokenizer的特殊token信息
    print("\n特殊token信息:")
    special_tokens = {
        "PAD": tokenizer.pad_token_id,
        "EOS": tokenizer.eos_token_id,
        "BOS": tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else None,
        "UNK": tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else None,
        "CLS": tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else None,
        "SEP": tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') else None,
        "MASK": tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else None
    }
    
    for name, token_id in special_tokens.items():
        if token_id is not None:
            try:
                token_str = tokenizer.decode([token_id])
                print(f"  {name} token (ID: {token_id}): '{token_str}'")
            except:
                print(f"  {name} token (ID: {token_id}): 无法解码")
    
    # 测试各种判断空输入的方法
    print("\n3. 各种判断空输入的方法比较:")
    
    # 方法1: 使用input_ids.sum(dim=1) == 0
    like_empty_sum = (like_encodings['input_ids'].sum(dim=1) == 0)
    dislike_empty_sum = (dislike_encodings['input_ids'].sum(dim=1) == 0)
    
    # 方法2: 使用注意力掩码总和为0或1
    like_empty_mask = (like_encodings['attention_mask'].sum(dim=1) <= 1)
    dislike_empty_mask = (dislike_encodings['attention_mask'].sum(dim=1) <= 1)
    
    # 方法3: 检查是否只有少量token
    like_empty_few = (like_encodings['attention_mask'].sum(dim=1) <= 2)
    dislike_empty_few = (dislike_encodings['attention_mask'].sum(dim=1) <= 2)
    
    # 预期的空输入判断结果
    expected_empty_like = [False, True, False, True]
    expected_empty_dislike = [True, False, False, True]
    
    print("方法1 (input_ids.sum(dim=1) == 0):")
    print(f"  like空检测结果: {like_empty_sum.tolist()}")
    print(f"  dislike空检测结果: {dislike_empty_sum.tolist()}")
    print(f"  与预期like匹配: {[a == b for a, b in zip(like_empty_sum.tolist(), expected_empty_like)]}")
    print(f"  与预期dislike匹配: {[a == b for a, b in zip(dislike_empty_sum.tolist(), expected_empty_dislike)]}")
    
    print("\n方法2 (attention_mask.sum(dim=1) <= 1):")
    print(f"  like空检测结果: {like_empty_mask.tolist()}")
    print(f"  dislike空检测结果: {dislike_empty_mask.tolist()}")
    print(f"  与预期like匹配: {[a == b for a, b in zip(like_empty_mask.tolist(), expected_empty_like)]}")
    print(f"  与预期dislike匹配: {[a == b for a, b in zip(dislike_empty_mask.tolist(), expected_empty_dislike)]}")
    
    print("\n方法3 (attention_mask.sum(dim=1) <= 2):")
    print(f"  like空检测结果: {like_empty_few.tolist()}")
    print(f"  dislike空检测结果: {dislike_empty_few.tolist()}")
    print(f"  与预期like匹配: {[a == b for a, b in zip(like_empty_few.tolist(), expected_empty_like)]}")
    print(f"  与预期dislike匹配: {[a == b for a, b in zip(dislike_empty_few.tolist(), expected_empty_dislike)]}")
    
    # 结论
    print("\n4. 结论:")
    methods = [
        ("input_ids.sum(dim=1) == 0", like_empty_sum.tolist(), dislike_empty_sum.tolist()),
        ("attention_mask.sum(dim=1) <= 1", like_empty_mask.tolist(), dislike_empty_mask.tolist()),
        ("attention_mask.sum(dim=1) <= 2", like_empty_few.tolist(), dislike_empty_few.tolist())
    ]
    
    for name, like_result, dislike_result in methods:
        like_matches = sum(1 for a, b in zip(like_result, expected_empty_like) if a == b)
        dislike_matches = sum(1 for a, b in zip(dislike_result, expected_empty_dislike) if a == b)
        total_matches = like_matches + dislike_matches
        print(f"方法 '{name}' 匹配度: like={like_matches}/{len(expected_empty_like)}, "
              f"dislike={dislike_matches}/{len(expected_empty_dislike)}, "
              f"总计={total_matches}/{len(expected_empty_like)+len(expected_empty_dislike)}")
    
    best_method = max(methods, key=lambda x: sum(1 for a, b in zip(x[1], expected_empty_like) if a == b) + 
                                          sum(1 for a, b in zip(x[2], expected_empty_dislike) if a == b))
    
    print(f"\n最佳检测空输入的方法是: '{best_method[0]}'")
    print("\n根据测试结果，建议在您的模型中使用以下判断空输入的方法:")
    print(f"like_is_empty = (like_attention_mask.sum(dim=1) <= 1)  # 查看有效token数量是否少于等于1")
    print(f"dislike_is_empty = (dislike_attention_mask.sum(dim=1) <= 1)  # 查看有效token数量是否少于等于1")

if __name__ == "__main__":
    test_empty_string_handling()