from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoModel
import click
from typing import TypeVar, Iterable, List
import torch
from tqdm import tqdm

"""
Task type refers to specific language direction

CUDA_VISIBLE_DEVICES=3,4 nohup python3 code/load_models.py -model_name deepseek_moe -task_type zh-en -batch_size 1 -eval_name wmt_sys > deepseek_moe_wmt_sys_eval_zh-en.out 2>&1 &
"""

T = TypeVar('T')
def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch  

def calculate_nll(cur_prompt, cur_gen, model, tokenizer):
    tokd_all = tokenizer(cur_gen, return_tensors='pt').input_ids.to(model.device)
    tokd_gen = tokenizer(cur_prompt, return_tensors='pt').input_ids.to(model.device)
    tokd_labels = tokd_all.clone().detach()
    tokd_labels[:, :tokd_labels.shape[1] - tokd_gen.shape[1] + 1] = -100
    with torch.no_grad():
        outputs = model(tokd_all, labels=tokd_labels)
        loss = outputs.loss
        #ppl = torch.exp(loss)
    return -loss.item()

name_dict = {'vicuna': 'lmsys/vicuna-7b-v1.5', 'llama': 'yahma/llama-7b-hf', 'llama2': 'meta-llama/Llama-2-7b-chat-hf',\
             'deepseek': 'deepseek-ai/deepseek-llm-7b-chat', 'deepseek_moe': "deepseek-ai/deepseek-moe-16b-chat", \
             'gpt-neox': 'EleutherAI/gpt-neox-20b', 'gpt-j': "EleutherAI/gpt-j-6b", 'mistral': 'mistralai/Mistral-7B-Instruct-v0.2', \
             'mistral_moe': 'mistralai/Mixtral-8x7B-Instruct-v0.1', "alpaca": "alpaca"}

@click.command()
@click.option('-model_name', help="alpaca")
@click.option('-task_type', help="zh-en, en-de or yor-en")
@click.option('-batch_size', help="8", type=int)
@click.option('-eval_name', help="eval file name", default=None)
def main(model_name, task_type, batch_size, eval_name):
    model = AutoModelForCausalLM.from_pretrained(name_dict[model_name], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(name_dict[model_name])
    tokenizer.pad_token = tokenizer.eos_token
    if task_type == "zh-en":
        src_lang="Chinese"
        tgt_lang="English"
    elif task_type == "en-de":
        src_lang="English"
        tgt_lang="German"
    elif task_type == "yor-en":
        src_lang="Yorba"
        tgt_lang="English"
    else:
        print("Your task is not supported!")
        exit(1)

    if eval_name == "wmt_sys":
        srcs = open(f'model_outputs/wmt_sys/{task_type}_src_wmt_sys.txt', 'r').readlines()
    else:
        srcs = open(f'srcs/{task_type}_src_100.txt', 'r').readlines()

    if eval_name:
        out_lines = open(f'model_outputs/{eval_name}/{task_type}_base_outputs_{eval_name}.txt', 'r').readlines()
        out_lines = [ele[:-1] for ele in out_lines]
        out_batch_ls = [list(ele) for ele in batchify(out_lines, batch_size)]
        f = open(f'model_outputs/{model_name}/{task_type}_{model_name}_eval_{eval_name}.txt', 'w')
    else:
        f = open(f'model_outputs/{model_name}/{task_type}_base_outputs_{model_name}.txt', 'w')
    
    with tqdm(total=len(srcs)) as pbar:
        for index, src_batch in enumerate(batchify(srcs, batch_size)):
            if model_name == "alpaca" or model_name == "vicuna":
                input_batch= [(
                    "Below is an instruction that describes a task. "
                    f"Translate {src_lang} text into {tgt_lang}.\n\n"
                    f"### Instruction:\n\n{src_lang}: {src[:-1]}\n\n### {tgt_lang}:"
                ) for src in src_batch]
            elif model_name == "llama2" or model_name == "gpt-j" or model_name == "gpt-neox":
                input_batch=[(
                    "Below is an instruction that describes a task. "
                    f"Translate Chinese text into English.\n\n"
                    f"### Instruction:\n\nChinese: 新华时评：把优秀返乡农民工打造成乡村振兴生力军-新华网\n\n### English: Xinhua Commentary: Outstanding returning rural migrant workers can be a rural revitalization army - Xinhuanet\n\n"
                    "Below is an instruction that describes a task. "
                    f"Translate English text into German.\n\n"
                    f"### Instruction:\n\nEnglish: You can come back any time as our chat service window is open 24/7\n\n### German: Sie können jederzeit wiederkommen, da unser Chat-Service-Fenster täglich rund um die Uhr geöffnet ist\n\n"
                    "Below is an instruction that describes a task. "
                    f"Translate Yorba text into English.\n\n"
                    f"### Instruction:\n\nYorba: Won da Olori Skwodroni. Dilokrit Pattavee gege bi awako ofururu.\n\n### English: The pilot was identified as Squadron Leader Dilokrit Pattavee.\n\n"
                    "Below is an instruction that describes a task. "
                    f"Translate {src_lang} text into {tgt_lang}.\n\n"
                    f"### Instruction:\n\n{src_lang}: {src[:-1]}\n\n### {tgt_lang}:"
                ) for src in src_batch]
            elif model_name == "deepseek" or model_name == "mistral" or model_name == "mistral_moe" or model_name == "deepseek_moe":
                model.generation_config = GenerationConfig.from_pretrained(name_dict[model_name])
                model.generation_config.pad_token_id = model.generation_config.eos_token_id
                input_batch_icl = [{"role": "user", "content": "Translate Chinesse text into English. Chinese: 新华时评：把优秀返乡农民工打造成乡村振兴生力军-新华网 English:"},\
                                    {"role": "assistant", "content": "Xinhua Commentary: Outstanding returning rural migrant workers can be a rural revitalization army - Xinhuanet"},\
                                    {"role": "user", "content": "Translate English text into German. English: You can come back any time as our chat service window is open 24/7 German:"},\
                                    {"role": "assistant", "content": "Sie können jederzeit wiederkommen, da unser Chat-Service-Fenster täglich rund um die Uhr geöffnet ist"},\
                                    {"role": "user", "content": "Translate Yorba text into English. Yorba: Won da Olori Skwodroni. Dilokrit Pattavee gege bi awako ofururu. English:"},\
                                    {"role": "assistant", "content": "The pilot was identified as Squadron Leader Dilokrit Pattavee."}]
                input_batch = [input_batch_icl + [{"role": "user", "content": f"Translate {src_lang} text into {tgt_lang}. {src_lang}: {src[:-1]} {tgt_lang}:"}] for src in src_batch]
                input_batch = [tokenizer.apply_chat_template(ele, add_generation_prompt=True, tokenize=False) for ele in input_batch] 
            else:
                print("We do not support other open sourced LLM at the moment!")
                exit(1)
            
            if eval_name:
                ppl = calculate_nll(input_batch, out_batch_ls[index], model, tokenizer)
                f.write(str(ppl)+'\n')
            else:
                inputs = tokenizer(input_batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
                out = model.generate(inputs=inputs.input_ids, max_new_tokens=256)
                output_text = tokenizer.batch_decode(out, skip_special_tokens=True)
            
                if model_name == "llama2" or model_name == "gpt-j" or model_name == "gpt-neox":
                    out_ls = [out.replace(inp, '').split('\n')[0].strip()+'\n' for inp, out in zip(input_batch, output_text)]
                elif model_name == "deepseek":
                    out_ls = [out.split('Assistant:')[-1].split('\n')[0].strip()+'\n' for out in output_text]
                elif model_name == "mistral" or model_name == "deepseek_moe" or model_name == "mistral_moe":
                    out_ls = [out.split('[/INST]')[4].split('\n')[0].strip()+'\n' for out in output_text]
                else:
                    out_ls = [out.replace(inp, '').replace('\n','').strip()+'\n' for inp, out in zip(input_batch, output_text)]
                
                # print(out_ls)
                for out in out_ls:
                    f.write(out)
            pbar.update(batch_size)

if __name__ == "__main__":
    main()