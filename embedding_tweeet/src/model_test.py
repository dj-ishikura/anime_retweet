import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# テストする入力テキスト
input_texts = [
    'query: 今日の天気は？',
    'query: 南瓜的家常做法',
    # 類似の質問を追加して検証
    'query: 明日の天気は？',
    'query: 南瓜怎么做好吃'  # "かぼちゃをおいしく作るには？"（中国語）
]

# モデルとトークナイザーの初期化
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

# 埋め込みの生成
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# L2正規化を適用
normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

# 結果の確認
print("基本情報:")
print(f"埋め込みベクトルの形状: {embeddings.shape}")  # (文章数, 次元数)
print(f"各ベクトルのノルム（正規化前）: {torch.norm(embeddings, dim=1).tolist()}\n")

# コサイン類似度の計算
similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())

print("コサイン類似度行列:")
for i, text1 in enumerate(input_texts):
    print(f"\n{text1}")
    for j, text2 in enumerate(input_texts):
        if i != j:  # 同じテキスト同士の比較を除外
            similarity = similarity_matrix[i][j].item()
            print(f"vs {text2}: {similarity:.3f}")

# ベクトルの一部を可視化
print("\n各テキストのベクトル（最初の5次元）:")
for i, text in enumerate(input_texts):
    print(f"\n{text}:")
    print(normalized_embeddings[i, :5].tolist())