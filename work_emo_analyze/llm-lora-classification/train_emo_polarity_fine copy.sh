echo "a"

# パラメータの配列
batch_sizes=(32 64 128)
epochs=(2 3 4)
learning_rates=("1e-5" "2e-5" "5e-5")

# 出力ディレクトリの親ディレクトリ
output_parent_dir="/work/n213304/learn/anime_retweet_2/work_emo_analyze/llm-lora-classification/outputs/rinna__japanese-gpt-neox-3.6b"

# 各パラメータの組み合わせごとにジョブを実行
for batch_size in "${batch_sizes[@]}"
do
    for epoch in "${epochs[@]}"
    do
        for lr in "${learning_rates[@]}"
        do
            # 出力ディレクトリの生成
            output_dir="${output_parent_dir}/bs_${batch_size}_ep_${epoch}_lr_${lr}"
            mkdir -p $output_dir

            # 新しいPBSジョブをサブミット
            qsub -v train_emo_polarity_fine_jobs.sh $batch_size $epoch $lr $output_dir
        done
    done
done

# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity.py --model_name rinna/japanese-gpt-neox-3.6b