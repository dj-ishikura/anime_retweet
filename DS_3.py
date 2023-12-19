# t の値と bigram 言語モデルの初期化
t = {
    ("gengo", "natural"): 1/2,
    ("shori", "natural"): 0,
    ("shizen", "natural"): 1/2,
    ("gengo", "language"): 2/3,
    ("shori", "language"): 1/6,
    ("shizen", "language"): 1/6,
    ("gengo", "processing"): 1/2,
    ("shori", "processing"): 1/2,
    ("shizen", "processing"): 0
}

bigram_language_model = {
    "SOS natural language EOS": 1/6,
    "SOS natural language processing EOS": 1/6,
    "SOS natural processing EOS": 1/3,
    "SOS language processing EOS": 1/6,
    "SOS language EOS": 1/6
}

# 翻訳元の文 J
J = ["gengo", "shori", "shizen"]

# 翻訳先の候補文 E のリスト
E_candidates = [
    ["natural", "language"],
    ["natural", "language", "processing"],
    ["natural", "processing"],
    ["language", "processing"],
    ["language"]
]

# 最も確率の高い翻訳を求める
best_translation = None
highest_probability = 0

for E in E_candidates:
    print(f"\nCalculating for E = {' '.join(E)}:")

    # L^M
    L = len(E)
    M = len(J)
    lm = 1 / (L ** M)
    print(f"1/{L}^{M} = {lm}")

    # P(J|E) の計算
    p_j_given_e = lm
    calculation_steps = []  # 途中計算を格納するリスト
    for j in J:
        sum_t = sum(t.get((j, e), 0) for e in E)
        p_j_given_e *= sum_t
        calculation_steps.append("(")
        for e in E:
            calculation_steps.append(f"t({j} | {e}) + ")
        calculation_steps.append(")")

    # 途中計算の表示
    print("".join(calculation_steps))
    print(f"P(J|E) = {p_j_given_e}")

    # bigram 言語モデルの確率を掛け合わせる
    lm_probability = bigram_language_model.get("SOS " + " ".join(E) + " EOS", 0)
    combined_probability = p_j_given_e * lm_probability
    print(f"Bigram language model probability: {lm_probability}")
    print(f"Combined probability: {combined_probability}")

    # 最も高い確率を持つ翻訳を更新
    if combined_probability > highest_probability:
        highest_probability = combined_probability
        best_translation = E

# 結果の表示
print(f"\nBest translation: {' '.join(best_translation)} with probability {highest_probability}")
