# 初期化
'''
english_words = ["natural", "language", "processing"]
japanese_words = ["shizen", "gengo", "syori"]
'''
english_words = ["b", "c"]
japanese_words = ["x", "y"]

# t(j|e) の初期値を設定（全て1）
translation_prob = {e: {j: 1 / len(japanese_words) for j in japanese_words} for e in english_words}
print(f'translation_prob : \n {translation_prob}')

# コーパス
'''
corpus = [
    (["natural", "language"], ["shizen", "gengo"]),
    (["language", "processing"], ["gengo", "syori"])
]
'''
corpus = [
    (["b", "c"], ["x", "y"]),
    (["b", "c"], ["y", "x"]),
    (["b"], ["y"]),
]

# p(e|j) 配列の初期化
p_afe = {}
for source_words, target_words in corpus:
    for source, target in zip(source_words, target_words):
        # キーが存在しない場合は新しい辞書を割り当てる
        if target not in p_afe:
            p_afe[target] = {}

        p_afe[target][source] = 1.0

print(p_afe)

# EMトレーニングの実施（2回の繰り返し）
for iteration in range(2):
    print(f"Iteration {iteration + 1}")

    # Eステップ：期待値の計算
    counts = {e: {j: 0.0 for j in japanese_words} for e in english_words}
    total = {e: 0.0 for e in english_words}
    
    i = 0
    for english, japanese in corpus:
        # この文ペアにおける確率の合計
        s_total = {e: sum(translation_prob[e][j] for j in japanese) for e in english}
        print(f's_total : \n {s_total}')

        # 各単語のカウントと合計を更新

        for e, j in zip(english, japanese):
            p_ej[i] *= translation_prob[e][j]
            counts[e][j] += translation_prob[e][j]
            total[e] += translation_prob[e][j] / s_total[e]
        i += 1
        print(f'p_ej : \n {p_ej}')           
    print(f'counts : \n {counts}')
    print(f'total : \n {total}')

    # Mステップ：パラメータの更新
    for e in english_words:
        for j in japanese_words:
            translation_prob[e][j] = counts[e][j] / total[e]

    # 途中結果の表示
    print("Translation Probabilities:")
    for e in english_words:
        for j in japanese_words:
            print(f"t({j}|{e}) = {translation_prob[e][j]}")
    print("\n")
