from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# tokenizer = AutoTokenizer.from_pretrained("google/madlad400-10b-mt")
# model = AutoModelForSeq2SeqLM.from_pretrained("google/madlad400-10b-mt", torch_dtype=torch.bfloat16, device_map="auto")
# src_lines = open('srcs/yor-en_src_100.txt', 'r').readlines()
# out_lines = []

# for src in src_lines:
#     batched_input = ["<2en> "+src[:-1]]
#     inputs = tokenizer(batched_input, return_tensors="pt", padding = True, max_length=128).to(model.device)

#     translated_tokens = model.generate(
#         **inputs, max_new_tokens=128
#     )
#     out_lines+=[tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]+'\n']

# with open('model_outputs/madlad400-10b-mt/yor-en_100.txt', 'w') as f:
#     f.writelines(out_lines)

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-moe-54b")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-moe-54b", torch_dtype=torch.bfloat16, device_map="auto")
src_lines = open('srcs/yor-en_src_100.txt', 'r').readlines()
out_lines = []

for src in src_lines:
    batched_input = [src[:-1]]
    inputs = tokenizer(batched_input, return_tensors="pt", padding = True, max_length=128).to(model.device)

    translated_tokens = model.generate(
        **inputs, max_new_tokens=128, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"]
    )
    out_lines+=[tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]+'\n']

with open('model_outputs/nllb-moe-54b/yor-en_100.txt', 'w') as f:
    f.writelines(out_lines)

