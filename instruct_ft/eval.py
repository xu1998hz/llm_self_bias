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

max_length = 2048
padding_strategy = "left"
batch_size = 4

# weight_path = 'ckpt/llama2_chat_ref_wmt21_zh-en_all_error/checkpoint-445'
weight_path = '/mnt/taurus/home/guangleizhu/peril_self_improve/instruct_ft/ckpt/llama2_chat_ift_mqm/checkpoint-935'
print(weight_path)
model_name = 'meta-llama/Llama-2-7b-chat-hf'
# model_name = 'mistralai/Mistral-7B-Instruct-v0.2'

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

src = '/home/guangleizhu/peril_self_improve/srcs/zh-en_src_100.txt'
src = open(src, 'r').readlines()
ref = '/home/guangleizhu/peril_self_improve/refs/zh-en_ref_100.txt'
ref = open(ref, 'r').readlines()

instruction_str = f"You are an annotator for the quality of machine translation. Your task is to identify errors and assess the quality of the translation.\nBased on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), locale convention (currency, date, name, telephone, or time format) style (awkward), terminology (inappropriate  for context, inconsistent use), non-translation, other, or no-error.\nEach error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension."
in_context_txt = f"""Source: ```大众点评乌鲁木齐家居商场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评``` Translation: ```Urumqi Home Furnishing Store Channel provides you with the latest bussiness information such as the address, telephone number, bussiness hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.``` Annotate errors in the translation. MQM annotations: "of high-speed rail" is a critical accuracy/addition error\n"go to the reviews" is a major accuracy/mistranslation error\n"etc.," is a minor style/awkwards error\n\nSource: ```I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.``` Translation: ```Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.``` Annotate errors in the translation. MQM annotations: 'involvement' is a major accuracy/mistranslation error\n'the account holder' is a major accuracy/omission error\n'wäre' is a minor fluency/grammar error\n'dir' is a minor fluency/register error\n\nSource: ```Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.``` Translation: ```Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemže obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.``` Annotate errors in the translation. MQM annotations: 've Vídni' is a major accuracy/addition error\n'the stop-start' is a major accuracy/omission error\n'partaje' is a minor terminology/inappropriate for context error\n\n"""
prefix = instruction_str + ' ' + in_context_txt

model.eval()
for i in tqdm(range(10)):
    batch = []
    for j in range(batch_size):
        idx = i * batch_size + j
        template = prefix + f""" Source: ```{src[idx][:-1]}``` Translation: ```{ref[idx][:-1]}``` Annotate errors in the translation. MQM annotations:"""
        # print(template)
        batch.append(template)
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = inputs.input_ids
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
        output = tokenizer.decode(outputs[i], skip_special_tokens=True) 
        # remove the prefix
        output = output[len(prefix):]
        print(output)
        print('-' * 60)