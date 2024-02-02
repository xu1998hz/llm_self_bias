#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --gpus=3
#SBATCH --time=10:00:00
#SBATCH --account=guangleizhu
#SBATCH --partition=taurus
#SBATCH --output=/home/guangleizhu/reproduce_pinpoint/slurm_out/ft_mistral.out
#SBATCH --error=/home/guangleizhu/reproduce_pinpoint/slurm_out/ft_mistral_error.out

source ~/.bashrc
# module purge
eval "$(conda shell.bash hook)"
conda activate torch2.1

nvidia-smi

# move to working directory
# this job assumes:
# - all input data is stored in this directory
# - all output should be stored in this directory
# - please note that groupname should be replaced by your groupname
# - username should be replaced by your username
# - path-to-directory should be replaced by the path to your directory where the executable is

# cd ../finetune

deepspeed --num_gpus 3 finetune_mistral.py --run_name mistral_ft_test
# deepspeed --num_gpus 4 finetune_llama.py --lang en-de