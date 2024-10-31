import json
import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from typing import List, Dict, Tuple
import numpy as np
import unicodedata
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os

class TweetDataset(Dataset):
    def __init__(self, texts: List[str], ids: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.ids[idx]

class TweetPreprocessor:
    def __init__(self):
        self.url_pattern = re.compile(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+')
        self.mention_pattern = re.compile(r'@[\w_]+')
        self.rt_pattern = re.compile(r'^RT\s+')
        self.html_pattern = re.compile(r'&[a-z]+;')

    def preprocess(self, text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        text = self.html_pattern.sub(' ', text)
        text = self.rt_pattern.sub('', text)
        text = self.mention_pattern.sub('', text)
        text = self.url_pattern.sub('', text)
        return text

class TweetEmbedding:
    def __init__(self, model_name='intfloat/multilingual-e5-large'):
        print("モデルの初期化中...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 利用可能なGPUの確認と設定
        self.device_ids = list(range(torch.cuda.device_count()))
        if not self.device_ids:
            raise RuntimeError("GPUが見つかりません")
        print(f"利用可能なGPU数: {len(self.device_ids)}")
        
        # メインGPUの設定
        self.device = f'cuda:{self.device_ids[0]}'
        
        # モデルの初期化とGPUへの転送
        self.model = AutoModel.from_pretrained(model_name)
        self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
        self.model = self.model.to(self.device)
        
        self.preprocessor = TweetPreprocessor()
        print("モデルの初期化完了")

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def process_batch(self, batch_texts: List[str], batch_ids: List[str]) -> Tuple[Tensor, List[str]]:
        # バッチのトークン化
        batch_dict = self.tokenizer(batch_texts, max_length=512, padding=True, 
                                  truncation=True, return_tensors='pt')
        
        # データをGPUに転送
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        
        # 推論
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        return embeddings, batch_ids

    def get_embeddings(self, tweets: List[Dict], batch_size: int = 64) -> Tuple[np.ndarray, List[str], List[str]]:
        """ツイートから埋め込みベクトルを生成（並列処理）"""
        print("テキストの前処理中...")
        texts = []
        ids = []
        for tweet in tqdm(tweets, desc="前処理"):
            processed = f"query: {self.preprocessor.preprocess(tweet['text'])}"
            if len(processed.strip()) > 7:
                texts.append(processed)
                ids.append(tweet['tweet_id'])

        # データセットとデータローダーの作成
        dataset = TweetDataset(texts, ids, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # バッチ処理で埋め込みを生成
        all_embeddings = []
        all_ids = []
        
        print(f"\n埋め込みベクトルの生成中... (バッチサイズ: {batch_size})")
        for batch_texts, batch_ids in tqdm(dataloader, desc="埋め込み生成"):
            embeddings, ids = self.process_batch(batch_texts, batch_ids)
            all_embeddings.append(embeddings.cpu().numpy())
            all_ids.extend(ids)

        # 結果の結合
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        
        return final_embeddings, texts, all_ids

def main(input_path, output_path):
    print(f"入力ファイル: {input_path}")
    print(f"出力ファイル: {output_path}")
    
    # ファイルからツイートを読み込む
    print("\nツイートの読み込み中...")
    tweets = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="ファイル読み込み"):
            tweets.append(json.loads(line))
    
    print(f"\n読み込んだツイート数: {len(tweets)}")
    
    # 埋め込みモデルの初期化
    embedder = TweetEmbedding()
    
    # 埋め込みの生成
    print("\n埋め込みベクトルの生成を開始...")
    embeddings, processed_texts, tweet_ids = embedder.get_embeddings(tweets)
    
    print(f"\n処理されたツイート数: {len(tweet_ids)}")
    print(f"埋め込みの形状: {embeddings.shape}")
    
    # 結果の保存
    print("\nデータの保存中...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, 
             embeddings=embeddings,
             tweet_ids=np.array(tweet_ids),
             processed_texts=np.array(processed_texts))
    
    print(f"\nデータを{output_path}に保存しました")
    print("処理完了!")

if __name__ == "__main__":
    input_path = '/work/n213304/learn/anime_retweet_2/random_stream_tweets_and_profile_2022_7-9.jsonl'
    output_path = '/work/n213304/learn/anime_retweet_2/embedding_tweeet/processed_data/tweet_embeddings.npz'
    main(input_path, output_path)