#PBS -q gLrchq
#PBS -l select=1:ncpus=4:mem=64G:ngpus=2
#PBS -v SINGULARITY_IMAGE=imc.tut.ac.jp/transformers-pytorch-cuda118:4.37.2
#PBS -k doe -j oe

cd ${PBS_O_WORKDIR}

TORCH_HOME=`pwd`/.cache/torch
TRANSFORMERS_CACHE=`pwd`/.cache/transformers
HF_HOME=`pwd`/.cache/huggingface
export TORCH_HOME TRANSFORMERS_CACHE HF_HOME
export TORCH_USE_CUDA_DSA=1

poetry run python src/embedding_tweet.py
