from transformers import LlamaForCausalLM, AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer
import torch
import json
from datasets import load_dataset
from transformers import AutoConfig, Trainer, TrainingArguments
import copy
from typing import Dict, Sequence
import transformers
import random
from tqdm import tqdm

max_length = 720
padding_strategy = "left"
batch_size = 8

# weight_path = 'ckpt/llama2_chat_ref_wmt21_zh-en_all_error/checkpoint-445'
weight_path = '/mnt/taurus/home/guangleizhu/peril_self_improve/instruct_ft/ckpt/mistral_inst_ift_mqm/checkpoint-935'
print(weight_path)
# model_name = 'meta-llama/Llama-2-7b-chat-hf'
model_name = 'mistralai/Mistral-7B-Instruct-v0.2'

model = AutoModelForCausalLM.from_pretrained(weight_path)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name,
    model_max_length=max_length,
    padding_side=padding_strategy,
    use_fast=False,
)
tokenizer.pad_token = tokenizer.unk_token

# data = 'data/wmt21_ref_train_all_error.json'
data = 'data/mqm_newstest2021_zh-en_train.json'
# data = 'data/ift_seed.json'
print(data)
with open(data, 'r') as f:
    data = json.load(f)
data_dict = data['instances']


model.eval()
for _ in tqdm(range(10)):
    batch = []
    for _ in range(batch_size):
        i = random.randint(0, len(data_dict))
        prompt = data_dict[i]['input']
        batch.append(prompt)
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = inputs.input_ids
    print(input_ids.shape)
    attention_mask = inputs.attention_mask
    outputs = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        max_new_tokens=512, 
        pad_token_id=tokenizer.unk_token_id,
        num_return_sequences=1, 
        do_sample=False,
    )
    # remove the input part
    # output = outputs[0][input_ids.shape[-1]:]
    for i in range(batch_size):
        output = outputs[i]
        print(tokenizer.decode(output, skip_special_tokens=True))
        print('-' * 60)