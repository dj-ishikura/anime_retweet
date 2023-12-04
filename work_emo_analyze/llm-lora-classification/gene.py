# 指定されたパラメータの配列
batch_sizes = [32, 64, 128]
epochs = [2, 3, 4]
learning_rates = [1e-5, 2e-5, 5e-5]

# 各組み合わせに対するコマンド文字列を生成
commands = []
for batch_size in batch_sizes:
    for epoch in epochs:
        for lr in learning_rates:
            command = f"poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size {batch_size} --epochs {epoch} --lr {lr}"
            commands.append(command)

# コマンドの文字列を出力
commands_str = "\n".join(commands)
print(commands_str)
