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

weight_path = '/home/guangleizhu/peril_self_improve/instruct_ft/ckpt/llama2_chat_ref_zh-en/checkpoint-640'
# weight_path = '/mnt/taurus/home/guangleizhu/peril_self_improve/instruct_ft/ckpt/llama2_chat_ift_mqm/checkpoint-935'
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

data = '/home/guangleizhu/peril_self_improve/instruct_ft/data/wmt21_ref_train_all_error.json'
with open(data, 'r') as f:
    data = json.load(f)
data = data['instances']

# instruction_str = f"You are an annotator for the quality of machine translation. Your task is to identify errors and assess the quality of the translation.\nBased on the reference segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), locale convention (currency, date, name, telephone, or time format) style (awkward), terminology (inappropriate  for context, inconsistent use), non-translation, other, or no-error.\nEach error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension."
# in_context_txt = f"""Reference: ```Experts say the best time to fall asleep is between 22:00-23:00, and the best time to wake up is between 05:30-06:30.``` Translation: ```Experts say that the best bedtime is 22 to 23 hours, and the best time to get up is 5:30 to 6:30.``` Annotate errors in the translation. MQM annotations: 'bedtime' is a major style/awkward error. '22 to 23 hours' is a major accuracy/mistranslation error.\n\nReference: ```Over the past few months, total bank deposits have also continued to increase, there has been a series of IPO's, and market transactions are increasing.``` Translation: ```The total amount of bank deposits has also continued to increase in the past few months. There has been a series of public offerings in the stock market, and market transactions have repeatedly increased.``` Annotate errors in the translation. MQM annotations: 'repeatedly increased' is a minor style/awkward error.\n\nReference: ```July 26, (Ecns.cn): According to the website of the Embassy of the People's Republic of China in the UK, on the 25th, the spokesperson of the Embassy of the People's Republic of China in the UK responded to reporters' questions about erroneous statements about cyber attacks.``` Translation: ```Www.chinanews.com, July 26 - According to the news at the website of the Chinese Embassy in London, on the 25th day, the spokesman of the Chinese Embassy in London answered reporters' questions on the wrong remarks about cyber attacks.``` Annotate errors in the translation. MQM annotations: 'at' is a major fluency/grammar error. 'on the 25th day,' is a major locale convention/date format error.\n\n"""
in_context_txt = "You are evaluating a Machine translation task. The reference translation is 'After the outbreak of the COVID-19, the Tianshan District Federation of Trade Unions invested more than RMB 1.05 million, which were urgently allocated to the 23 subordinate grassroots trade unions.'. The model generated translation is 'After the outbreak of COVID-19 epidemic, Tianshan District Federation of Trade Unions invested more than RMB 1.05 million in the first time, which was urgently allocated to 23 directly affiliated grass-roots trade unions.'. Please identify all errors in the translation, up to a maximum of five. For each error, please give me the corresponding error location, error type and major/minor label for each error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed. 'in the first time' is a major accuracy/mistranslation error. 'urgently' is a major style/awkward error.\n\n"
# prefix = instruction_str + ' ' + in_context_txt

model.eval()
for i in tqdm(range(10)):
    batch = []
    for j in range(batch_size):
        idx = i * batch_size + j
        # template = prefix + f"""Reference: ```{data[idx]['ref']}``` Translation: ```{data[idx]['mt']}``` Annotate errors in the translation. MQM annotations:"""
        template = in_context_txt + data[idx]['input']
        # print(template)
        # exit()
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
        output = output[len(in_context_txt):]
        print(output)
        print('-' * 60)