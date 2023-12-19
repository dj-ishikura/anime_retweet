# コーパスのアライメント

corpus = [
    (["b", "c"], ["x", "y"]),
    (["b", "c"], ["y", "x"]),
    (["b"], ["y"])
]

corpus = [
    (["natural", "language"], ["shizen", "gengo"]),
    (["language", "processing"], ["gengo", "syori"])
]

source_words = {"natural", "language", "processing"}
target_words = {"shizen", "gengo", "syori"}
# source_words = {"b", "c"}
# target_words = {"x", "y"}

alignments = []
for source_sentence, target_sentence in corpus:
    key = "".join(source_sentence) + '-' + "".join(target_sentence)
    alignments.append((source_sentence, target_sentence, key))

print(alignments)

source_words = set(word for source_sentence, _, _ in alignments for word in source_sentence)
target_words = set(word for _, target_sentence, _ in alignments for word in target_sentence)
t = {(target, source): 1.0 / len(target_words) for source in source_words for target in target_words}

print(f"iteration 0 :")
for key, value in t.items():
    print(f"t({key[0]} | {key[1]}) = {value}")

for i in range(2):
    # Step 2: P(a,f | e) の計算
    # 各アライメントの P(a,f | e) を計算
    p_afe = {}
    for source_sentence, target_sentence in corpus:
        key = tuple(source_sentence), tuple(target_sentence)
        p_afe[key] = 1.0
        for e, j in zip(source_sentence, target_sentence):
            p_afe[key] *= t[(j, e)]
    print(f'p_afe : \n {p_afe}')

    # P(a,f | e) の合計を計算し、P(a | e,f) を求める
    p_aef = {}
    for key in p_afe:
        source_sentence, _ = key
        total_p_afe = sum(p_afe[k] for k in p_afe if k[0] == source_sentence)
        p_aef[key] = p_afe[key] / total_p_afe
    print(f'p_aef : \n {p_aef}')

    # Step 4: 分数カウントの集計
    tc = {}
    for source_sentence, target_sentence in corpus:
        for e, j in zip(source_sentence, target_sentence):
            key = (j, e)
            if key not in tc:
                tc[key] = 0
            tc[key] += p_aef[(tuple(source_sentence), tuple(target_sentence))]

    print(f'tc : \n {tc}')
    # Step 5: パラメータの更新
    # 分数カウントの集計
    for e in source_words:
        # 各ソース言語単語の総分数カウント
        total_tc = sum(tc[(j, e)] for j, e_source in tc if e_source == e)

        # 翻訳確率の更新
        for j in target_words:
            key = (j, e)
            if key in tc:  # tc 辞書にキーが存在する場合のみ処理
                t[key] = tc[key] / total_tc

    
    print(f"iteration {i+1} :")
    for key, value in t.items():
        print(f"t({key[0]} | {key[1]}) = {value}")

