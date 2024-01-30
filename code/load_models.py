from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import click
from typing import TypeVar, Iterable, List
from manifest import Manifest

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

name_dict = {'vicuna': 'lmsys/vicuna-7b-v1.5', 'llama': 'yahma/llama-7b-hf', 'llama2': 'meta-llama/Llama-2-7b-chat-hf',\
             'deepseek': 'deepseek-ai/deepseek-llm-7b-chat', 'deepseek_moe': "deepseek-ai/deepseek-moe-16b-chat", \
             'gpt-neox': 'EleutherAI/gpt-neox-20b', 'gpt-j': "EleutherAI/gpt-j-6b", 'mistral': 'mistralai/Mistral-7B-Instruct-v0.2', \
             'mistral_moe': 'mistralai/Mixtral-8x7B-Instruct-v0.1'}

@click.command()
@click.option('-model_name', help="alpaca")
@click.option('-task_type', help="zh-en, en-de or yor-en")
@click.option('-batch_size', help="8", type=int)
@click.option('-ip', help="http://172.31.11.128:5000")
def main(model_name, task_type, batch_size, ip):
    if model_name == "gpt-neox" or model_name == "deepseek_moe" or model_name == "mistral_moe":
        model = Manifest(
            client_name = "huggingface",
            client_connection = ip,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(name_dict[model_name], trust_remote_code=True).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(name_dict[model_name])
    tokenizer.pad_token = tokenizer.eos_token
    if task_type == "zh-en":
        srcs = open(f'srcs/zh-en_src_100.txt', 'r').readlines()
        src_lang="Chinese"
        tgt_lang="English"
    elif task_type == "en-de":
        srcs = open(f'srcs/en-de_src_100.txt', 'r').readlines()
        src_lang="English"
        tgt_lang="German"
    elif task_type == "yor-en":
        srcs = open(f'srcs/yor-en_src_100.txt', 'r').readlines()
        src_lang="Yorba"
        tgt_lang="English"
    else:
        print("Your task is not supported!")
        exit(1)
    
    out_ls = []
    for src_batch in batchify(srcs, batch_size):
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
        elif model_name == "deepseek" or model_name == "mistral" or model_name == "deepseek_moe" or model_name == "mistral_moe":
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
        
        # print(input_batch)
        # print('-'*20)
        
        if model_name == 'gpt-neox' or model_name == "deepseek_moe" or model_name == "mistral_moe":
            output_text = model.run(input_batch, n=1, max_new_tokens=512, do_sample=False)
            out_ls += [ele.split('\n')[0].strip()+'\n' for ele in output_text]
            # print(out_ls)
            # print('-'*50)
        else:
            inputs = tokenizer(input_batch, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            out = model.generate(inputs=inputs.input_ids.to("cuda"), max_new_tokens=256)
            output_text = tokenizer.batch_decode(out, skip_special_tokens=True)
        
            if model_name == "llama2" or model_name == "gpt-j":
                out_ls += [out.replace(inp, '').split('\n')[0].strip()+'\n' for inp, out in zip(input_batch, output_text)]
            elif model_name == "deepseek":
                out_ls += [out.split('Assistant:')[-1].split('\n')[0].strip() for out in output_text]
            elif model_name == "mistral":
                out_ls += [out.split('[/INST]')[-1].split('\n')[0].strip() for out in output_text]
            else:
                out_ls += [out.replace(inp, '').replace('\n','').strip()+'\n' for inp, out in zip(input_batch, output_text)]

        
    
    with open(f'{task_type}_base_outputs_{model_name}.txt', 'w') as f:
        f.writelines(out_ls)
    print(f"{task_type}_base_outputs_{model_name}.txt is saved!")

if __name__ == "__main__":
    main()