# OPENAI_API_KEY="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" python3 code/generate_base_output.py -lang_dir en-de -api_source openai  -model_type gpt-4
import click
from openai import OpenAI
from tqdm import tqdm
import backoff
from openai import RateLimitError
import google.generativeai as genai
import google.generativeai as palm
import time
from datasets import load_dataset
import glob
import json
from google.generativeai.types import safety_types
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoModel
import torch

genai.configure(api_key="AIzaSyD6TPDOsho_SsIGneOHNLjAyN07JCGnwyk")
palm.configure(api_key="AIzaSyD6TPDOsho_SsIGneOHNLjAyN07JCGnwyk")

name_dict = {'vicuna': 'lmsys/vicuna-7b-v1.5', 'llama': 'yahma/llama-7b-hf', 'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',\
             'deepseek': 'deepseek-ai/deepseek-llm-7b-chat', 'deepseek_moe': "deepseek-ai/deepseek-moe-16b-chat", 'mistral-base': 'mistralai/Mistral-7B-v0.1', \
             'gpt-neox': 'EleutherAI/gpt-neox-20b', 'gpt-j': "EleutherAI/gpt-j-6b", 'mistral-inst2': 'mistralai/Mistral-7B-Instruct-v0.2', \
             'mistral_moe': 'mistralai/Mixtral-8x7B-Instruct-v0.1', "alpaca": "alpaca", "llama2-70b": 'meta-llama/Llama-2-70b-chat-hf', \
             "llama2-13b": 'meta-llama/Llama-2-13b-chat-hf', "ift": "/mnt/taurus/home/guangleizhu/reproduce_pinpoint/finetune/ft_out/mistral_ft_test/checkpoint-66",\
             "eft": "/home/guangleizhu/peril_self_improve/instruct_ft/ckpt/mistral_eft/checkpoint-156", 'mistral-inst1': 'mistralai/Mistral-7B-Instruct-v0.1', \
             'mistral-inst1-mqm': "instruct_ft/ckpt/mistral_instruct_mqm_ift/checkpoint-187/", "mistral-inst2-mqm": "/home/guangleizhu/peril_self_improve/instruct_ft/ckpt/mistral_inst_ift_mqm/checkpoint-187"}

@backoff.on_exception(backoff.expo, RateLimitError)
def completions_with_backoff_openai(client, system_prompt, prompt_txt, model_type):
    response = client.chat.completions.create(
        model=model_type,  # "gpt-3.5-turbo", "gpt-4"
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt_txt},
        ],
        temperature=1.0,
        max_tokens=2048,
        top_p=1,
    )
    return response


@backoff.on_exception(backoff.expo, RateLimitError)
def completions_with_google(system_prompt, prompt_txt, model_type):
    if model_type == "gemini":
        model = genai.GenerativeModel(model_name="gemini-pro")
        completion = model.generate_content(
            system_prompt + prompt_txt,
            generation_config={"temperature": 1.0, "max_output_tokens": 1024},
        )
        try:
            return completion.text
        except:
            return "[BLOCKED]"
    elif model_type == "palm2":
        completion = palm.generate_text(
            model="models/text-bison-001",
            prompt=system_prompt + prompt_txt,
            temperature=1.0,
            max_output_tokens=2048,
            safety_settings=[
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_DEROGATORY,
                    "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_VIOLENCE,
                    "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_UNSPECIFIED,
                    "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_TOXICITY,
                    "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_SEXUAL,
                    "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
                    "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_DANGEROUS,
                    "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
            ],
        )
        if completion.result:
            return completion.result
        else:
            return "[BLOCKED]"
    else:
        print("model type is not supported!")
        exit(1)


@click.command()
@click.option("-task_type", help="mt, sci or code")
@click.option("-lang_dir")
@click.option("-api_source", default="google or openai")
@click.option(
    "-model_type", default="model name like gemini, palm2, gpt-3.5-turbo and gpt-4"
)
@click.option("-save_name")
def main(lang_dir, api_source, model_type, task_type, save_name):
    if api_source == "openai":
        client = OpenAI()
    elif api_source == "transformers":
        model = AutoModelForCausalLM.from_pretrained(name_dict[model_type], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(name_dict[model_type])
        tokenizer.pad_token = tokenizer.eos_token

    if task_type == "mt":
        if lang_dir == "zh-en":
            src_lang = "Chinese"
            tgt_lang = "English"
            src_lines = open("srcs/zh-en_src_100.txt", "r").readlines()
        elif lang_dir == "en-de":
            src_lang = "English"
            tgt_lang = "German"
            src_lines = open("srcs/en-de_src_100.txt", "r").readlines()
        elif lang_dir == "yor-en":
            src_lang = "Yoruba"
            tgt_lang = "English"
            src_lines = open("srcs/yor-en_src_100.txt", "r").readlines()
        else:
            print("Language direction is not supported!")
            exit(1)
        system_prompt = f"You are translating {src_lang}-to-{tgt_lang} machine translation. Do not provide any explanations or text apart from the translation. "
    elif task_type == "sci":
        system_prompt = "You are a scientific problem solver. Generate rationale in latex format and generate final answer after #### in python3 float format (only float number, no explantion). For example, ####0.1"
        src_lines = []
        for file_name in glob.glob("scibench/dataset/original/*_sol.json"):
            data = json.load(open(file_name))
            src_lines += [ele["problem_text"] for ele in data[1:]]
    else:
        print(f"{task_type} is not supported!")
        exit(1)

    out_ls = []
    for line in tqdm(src_lines):
        if task_type == "mt":
            if api_source == "transformers":
                prompt_txt = (
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
                    f"### Instruction:\n\n{src_lang}: {line[:-1]}\n\n### {tgt_lang}:"
                )
            else:
                prompt_txt = (
                    f"""{src_lang} source: \n{line[:-1]}\n{tgt_lang} translation:\n"""
                )
        elif task_type == "sci":
            icl_str = "Question: A one-particle, one-dimensional system has $\\Psi=a^{-1 / 2} e^{-|x| / a}$ at $t=0$, where $a=1.0000 \\mathrm{~nm}$. At $t=0$, the particle's position is measured.  (b) Find the probability that the measured value is between $x=0$ and $x=2 \\mathrm{~nm}$.\nRationale: (b) Use of Eq. (1.23) and $|x|=x$ for $x \\geq 0$ gives\r\n$$\r\n\\begin{aligned}\r\n\\operatorname{Pr}(0 \\leq x \\leq 2 \\mathrm{~nm}) & =\\int_0^{2 \\mathrm{~nm}}|\\Psi|^2 d x=a^{-1} \\int_0^{2 \\mathrm{~nm}} e^{-2 x / a} d x \\\\\r\n& =-\\left.\\frac{1}{2} e^{-2 x / a}\\right|_0 ^{2 \\mathrm{~nm}}=-\\frac{1}{2}\\left(e^{-4}-1\\right)=0.4908\r\n\\end{aligned}\r\n$$\n####0.4908"
            prompt_txt = f"{icl_str} Question: {line}\nRationale: "

        if api_source == "openai":
            response = (
                completions_with_backoff_openai(
                    client, system_prompt, prompt_txt, model_type
                )
                .choices[0]
                .message.content
            )
        elif api_source == "google":
            indicater = True
            while indicater:
                try:
                    response = completions_with_google(system_prompt, prompt_txt, model_type)
                    indicater = False
                except:
                    continue
        elif api_source == "transformers":
            inputs = tokenizer([prompt_txt], return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
            out = model.generate(inputs=inputs.input_ids, max_new_tokens=128)
            response = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
            if model_type == "mistral_moe":
                response = response.replace(prompt_txt, '').split('\n\n')[0].strip()
            else:
                response = response.split('### Instruction:')[4].split(f"### {tgt_lang}:")[1].split("\n\n")[0].strip()
        else:
            print("API source is not supported!")
            exit(1)

        out_ls += [response.replace("\n", "") + "\n"]

    with open(save_name, "w") as f:
        f.writelines(out_ls)
        print(f"{save_name} is saved!")

if __name__ == "__main__":
    main()
