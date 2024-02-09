#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --account=guangleizhu
#SBATCH --partition=gemini
#SBATCH --output=/home/guangleizhu/peril_self_improve/instruct_ft/slurm/ft_llama2_chat_ref.out
#SBATCH --error=/home/guangleizhu/peril_self_improve/instruct_ft/slurm/ft_llama2_chat_ref_error.out

source ~/.bashrc
# module purge
eval "$(conda shell.bash hook)"
conda activate torch2.1

# nvidia-smi

# move to working directory
# this job assumes:
# - all input data is stored in this directory
# - all output should be stored in this directory
# - please note that groupname should be replaced by your groupname
# - username should be replaced by your username
# - path-to-directory should be replaced by the path to your directory where the executable is

# cd ../finetune

deepspeed --num_gpus 4 finetune_llama2.py --run_name llama2_chat_ref_wmt21_zh-en_all_error
