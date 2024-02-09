import transformers
from mt_metrics_eval import data as mt_data
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import math
import argparse
from tqdm import tqdm
import json
import re
# from manifest import Manifest

argparse = argparse.ArgumentParser()
argparse.add_argument('--wmt', default='wmt22')
argparse.add_argument('--lang', default='zh-en')
argparse.add_argument('--batch_size', default=1)
args = argparse.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# weight_path = '/home/guangleizhu/reproduce_pinpoint/finetune/ft_out/mistral_ft_test/checkpoint-66'
# for aries
weight_path = '/mnt/taurus/home/guangleizhu/reproduce_pinpoint/finetune/ft_out/mistral_ft_test/checkpoint-266'

# load the fine tuned model
tokenizer = AutoTokenizer.from_pretrained(weight_path, use_fast=True)
# change max_length to 2k
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(weight_path, device_map=device, torch_dtype=torch.float32)

def check_ranking(x, y):
    assert len(x) == len(y)
    length = len(x)
    for i in range(length):
        for j in range(i + 1, length):
            # Compare the order of elements in x and y, allowing ties
            if (x[i] < x[j] and y[i] > y[j]) or (x[i] > x[j] and y[i] < y[j]):
                return False
    return True 

def get_score(cot):
    # find the score from cot of format "Score: <total points>"
    return 1

def predict(instruction, responses):
    prompts = []
    for r in responses:
        prompt = f"Review the user’s question and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:\n- Add 1 point if the response is relevant and provides some information related to the user’s inquiry, even if it is incomplete or contains some irrelevant content.\n- Add another point if the response addresses a substantial portion of the user’s question, but does not completely resolve the query or provide a direct answer.\n- Award a third point if the response answers the basic elements of the user’s question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.\n- Grant a fourth point if the response is clearly written from an AI Assistant’s perspective, addressing the user’s question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.\n- Bestow a fifth point for a response that is impeccably tailored to the user’s question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.\n\nUser: {instruction}\n\n<response>{r}</response>\n\nAfter examining the user’s instruction and the response:\n- Briefly justify your total score, up to 100 words.\n- Conclude with an integer score using the format: “Score: <total points>”\nRemember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary. To evaluate the response in alignment with this additive scoring model, we’ll systematically attribute points based on the outlined criteria."
        prompts.append(prompt)
    batch = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=2048).to(device)
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    model.eval()
    outputs = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        max_new_tokens=1024, 
        pad_token_id=tokenizer.unk_token_id,
        do_sample=False,
        # pad_token_id=tokenizer.eos_token_id,
        # temperature=0
    )

    out = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    input = [tokenizer.decode(input_ids[i], skip_special_tokens=True) for i in range(input_ids.shape[0])]
    cot = [o.replace(i, '').strip() for o, i in zip(out, input)]
    for i in range(len(cot)):
        print(cot[i])
        print('-' * 60)
    return cot, [0] * len(cot)


with open('data/tree_lst.json') as f:
    tree_lst = json.load(f)

target_num = 2306
current_num = 0

data_dict = []

print(f'there are total of {len(tree_lst)} conversation trees')

for tree in tqdm(tree_lst[:20]):
    target_scores = tree['response_quality']
    instruction = tree['instruction']
    cots, pred_scores = predict(instruction, tree['responses'])
    print('=' * 60)
    # if ranking is correct
    if check_ranking(target_scores, pred_scores):
        current_num += len(pred_scores)
        for i in range(len(pred_scores)):
            d = dict(
                instruction=tree['instruction'],
                response=tree['responses'][i],
                cot=cots[i],
                score=pred_scores[i]
            )
            data_dict.append(d)

    if current_num >= target_num:
        print('getting enough data!')
        break

with open('test_out.json', 'w') as f:
    json.dump(data_dict, f, indent=4)





