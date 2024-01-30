export FLASK_PORT=5000
CUDA_VISIBLE_DEVICES=0,1 nohup python3 -m manifest.api.app \
    --model_type huggingface \
    --model_name_or_path EleutherAI/gpt-neox-20b \
    --model_generation_type text-generation \
    --use_accelerate_multigpu \
    --percent_max_gpu_mem_reduction 0.85 > neox.out 2>&1 &

export FLASK_PORT=5001
CUDA_VISIBLE_DEVICES=2,3,4 nohup python3 -m manifest.api.app \
    --model_type huggingface \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --model_generation_type text-generation \
    --use_accelerate_multigpu \
    --percent_max_gpu_mem_reduction 0.85 > mistral_moe.out 2>&1 &


export FLASK_PORT=5002
CUDA_VISIBLE_DEVICES=5,6,7 nohup python3 -m manifest.api.app \
    --model_type huggingface \
    --model_name_or_path deepseek-ai/deepseek-moe-16b-chat \
    --model_generation_type text-generation \
    --use_accelerate_multigpu \
    --percent_max_gpu_mem_reduction 0.85 > deepseek_moe.out 2>&1 &