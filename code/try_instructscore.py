from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

model = AutoModelForCausalLM.from_pretrained('/mnt/gemini/home/guangleizhu/peril_self_improve/instruct_ft/ckpt/llama2_chat_ref_zh-en/checkpoint-128', torch_dtype=torch.bfloat16).to('cuda')
tokenizer = AutoTokenizer.from_pretrained('/mnt/gemini/home/guangleizhu/peril_self_improve/instruct_ft/ckpt/llama2_chat_ref_zh-en/checkpoint-128')
data = json.load(open('/mnt/gemini/home/guangleizhu/peril_self_improve/instruct_ft/data/mqm_newstest2021_zh-en_ref_train.json'))

for sys_prompt in data['instances']:
    inputs = tokenizer(sys_prompt['input'], return_tensors="pt").to('cuda')
    output = model.generate(inputs=inputs.input_ids, max_new_tokens=128)
    response = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(response)
    print()