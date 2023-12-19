import itertools

# 翻訳確率（先ほどのEMアルゴリズムの結果を使用）
translation_prob = {
    "natural": {"shizen": 0.5, "gengo": 0.5, "shori": 0.0},
    "language": {"shizen": 0.16666666666666666, "gengo": 0.6666666666666666, "shori": 0.16666666666666666},
    "processing": {"shizen": 0.0, "gengo": 0.5, "shori": 0.5}
}

# 対象となる文
english_sentence = ["natural", "language", "processing"]
japanese_sentence = ["shizen", "gengo", "shori"]

# P(J|E)の計算
prob_sum = 1.0
for j_word in japanese_sentence:
    print(f't({j_word}|{english_sentence[0]}) + t({j_word}|{english_sentence[1]}) + t({j_word}|{english_sentence[2]})')
    # 各日本語単語について、全ての英語単語との翻訳確率の合計を計算
    e = [translation_prob[e_word].get(j_word, 0) for e_word in english_sentence]
    print("*")
    prob_sum *= sum(translation_prob[e_word].get(j_word, 0) for e_word in english_sentence)

print(f"確率 P(J|E) = {prob_sum}")
