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
from typing import Set, Dict, TypeVar, Iterable, List
from google.generativeai.types import safety_types
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoModel
import torch
import jsonlines
import re
import pandas as pd
import nltk
import spacy
from datasets import load_dataset
from google.generativeai.types import HarmCategory, HarmBlockThreshold

T = TypeVar('T')

genai.configure(api_key="AIzaSyD6TPDOsho_SsIGneOHNLjAyN07JCGnwyk")
palm.configure(api_key="AIzaSyD6TPDOsho_SsIGneOHNLjAyN07JCGnwyk")

name_dict = {'vicuna': 'lmsys/vicuna-7b-v1.5', 'llama': 'yahma/llama-7b-hf', 'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',\
             'deepseek': 'deepseek-ai/deepseek-llm-7b-chat', 'deepseek_moe': "deepseek-ai/deepseek-moe-16b-chat", 'mistral-base': 'mistralai/Mistral-7B-v0.1', \
             'gpt-neox': 'EleutherAI/gpt-neox-20b', 'gpt-j': "EleutherAI/gpt-j-6b", 'mistral-inst2': 'mistralai/Mistral-7B-Instruct-v0.2', \
             'mistral_moe': 'mistralai/Mixtral-8x7B-Instruct-v0.1', "alpaca": "alpaca", "llama2-70b": 'meta-llama/Llama-2-70b-chat-hf', \
             "llama2-13b": 'meta-llama/Llama-2-13b-chat-hf', "ift": "/mnt/taurus/home/guangleizhu/reproduce_pinpoint/finetune/ft_out/mistral_ft_test/checkpoint-66",\
             "eft": "/home/guangleizhu/peril_self_improve/instruct_ft/ckpt/mistral_eft/checkpoint-156", 'mistral-inst1': 'mistralai/Mistral-7B-Instruct-v0.1', \
             'mistral-inst1-mqm': "instruct_ft/ckpt/mistral_instruct_mqm_ift/checkpoint-187/", "mistral-inst2-mqm": "/home/guangleizhu/peril_self_improve/instruct_ft/ckpt/mistral_inst_ift_mqm/checkpoint-187", \
             'mistral-inst1-mqm_fixed': "instruct_ft/ckpt/mistral_instruct_mqm_ift_fixed/checkpoint-187", "iter0": "UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0", "iter1": "UCLA-AGI/zephyr-7b-sft-full-SPIN-iter1", \
             "iter2": "UCLA-AGI/zephyr-7b-sft-full-SPIN-iter2", "iter3": "UCLA-AGI/zephyr-7b-sft-full-SPIN-iter3"}

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

@backoff.on_exception(backoff.expo, RateLimitError)
def completions_with_backoff_openai(client, system_prompt, prompt_txt, model_type,n):
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
        n=n,
    )
    return response


@backoff.on_exception(backoff.expo, RateLimitError)
def completions_with_google(system_prompt, prompt_txt, model_type):
    if model_type == "gemini":
        model = genai.GenerativeModel(model_name="gemini-pro")
        completion = model.generate_content(
            system_prompt + prompt_txt,
            generation_config={"temperature": 1.0, "max_output_tokens": 1024, "candidate_count": 1},
            safety_settings={
            # HarmCategory.HARM_CATEGORY_DANGEROUS: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_DEROGATORY: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_MEDICAL: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_SEXUAL: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_TOXICITY: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_VIOLENCE: HarmBlockThreshold.BLOCK_NONE
            }
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

nlp = spacy.load("en_core_web_sm")

def detect_concepts(sentence: str, concepts: List[str]) -> Set[str]:
    present_concepts = []
    
    # Tokenize the sentence and lemmatize the tokens
    tokens = nltk.word_tokenize(sentence)
    lemmas = [token.lemma_ for token in nlp(sentence)]
    
    # Check if each concept is present in the sentence
    for concept in concepts:
        if concept in tokens or concept in lemmas:
            present_concepts.append(concept)
    
    return set(present_concepts)

@click.command()
@click.option("-task_type", help="mt, sci or code")
@click.option("-lang_dir")
@click.option("-api_source", default="google or openai")
@click.option(
    "-model_type", default="model name like gemini, palm2, gpt-3.5-turbo and gpt-4"
)
@click.option("-save_name")
@click.option("-batch_size", type=int)
@click.option("-ret_seq", type=int, default=1)
def main(lang_dir, api_source, model_type, task_type, save_name, batch_size, ret_seq):
    if api_source == "openai":
        client = OpenAI()
    elif api_source == "transformers":
        tokenizer = AutoTokenizer.from_pretrained(name_dict[model_type])
        if tokenizer.pad_token is None:
            if model_type[:6] != 'llama2': 
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = "[PAD]"
                tokenizer.padding_side = "left"
                print(f"Padding token is not found, setting padding token to [PAD]")
        model = AutoModelForCausalLM.from_pretrained(name_dict[model_type], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

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
        elif lang_dir == "jav_Latn-en":
            src_lang = "Javanese"
            tgt_lang = "English"
            src_lines = open("srcs/jav_Latn-en_src_100.txt", "r").readlines()
        elif lang_dir == "hye_Armn-en":
            src_lang = "Armenian"
            tgt_lang = "English"
            src_lines = open("srcs/hye_Armn-en_src_100.txt", "r").readlines()
        elif lang_dir == "ibo_Latn-en":
            src_lang = "Igbo"
            tgt_lang = "English"
            src_lines = open("srcs/ibo_Latn-en_src_100.txt", "r").readlines()
        else:
            print("Language direction is not supported!")
            exit(1)
        system_prompt = f"You are translating {src_lang}-to-{tgt_lang} machine translation. Do not provide any explanations or text apart from the translation. "
    elif task_type == "commonsenseQA":
        # we select 200 samples from commonsense QA
        src_lines = [{'question': ele['question'], 'choices': ele['choices']} for ele in load_dataset('tau/commonsense_qa')['test']][:100]
    elif task_type == "commongen":
        system_prompt = "You are generating text based on specified words. Do not provide any explanations or text apart from the text output."
        src_lines = []
        with jsonlines.open('srcs/commongen_hard.jsonl') as reader:
            for line in reader:
                src_lines.append(line)
        src_lines = src_lines[:100]
    elif task_type == "math":
        system_prompt = "You are a competitive math problem solver. Please generate a step-by-step solution. Your final answer must be enclosed in LaTeX's \boxed tag."
        src_lines = load_dataset('hendrycks/competition_math')['test']['problem'][:100]
    else:
        print(f"{task_type} is not supported!")
        exit(1)

    out_ls = []
    with tqdm(total=len(src_lines)) as pbar:
        for batch_line in batchify(src_lines, batch_size):
            if task_type == "mt":
                if api_source == "transformers":
                    prompt_txt_ls = [(
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
                    ) for line in batch_line]
                else:
                    prompt_txt_ls = [(
                        f"""{src_lang} source: \n{line[:-1]}\n{tgt_lang} translation:\n"""
                    ) for line in batch_line]
            elif task_type == "commonsenseQA":
                icl_str = "Q: Sammy wanted to go to where the people were. Where might he go?\n\nAnswer Choices: A) race track, B) populated areas, C) the desert, D) apartment, E) roadblock \n\nExplain your reasoning. You must choose only one option from A to E. Your final answer should be a single letter from A to E, in the form (answer), at the end of your response.\n\n###A: Sammy would likely go to populated areas if he wants to be where the people are. Although there may be people in areas like a race track or an apartment, these are specific places that don't always guarantee the presence of people. Populated areas, on the other hand, are generally guaranteed to have people. The desert and a roadblock are also less likely areas for people to gather. So, the best answer is B) populated areas.\n\n(answer: B)"
                prompt_txt_ls = [f"{icl_str}\n\nQ: {line['question']}\n\nAnswer Choices: A) {line['choices']['text'][0]}, B) {line['choices']['text'][1]}, C) {line['choices']['text'][2]}, D) {line['choices']['text'][3]}, E) {line['choices']['text'][4]}\n\nExplain your reasoning. You must choose only one option from A to E. Your final answer should be a single letter from A to E, in the form (answer), at the end of your response.\n\n###A: " for line in batch_line]
            elif task_type == "commongen":
                prompt_txt_ls = [f"Please generate a sentence that contains the exact string matches for all the following concepts (All concepts must be used): \n{line['concepts']}" for line in batch_line]
            elif task_type == "math":
                icl_str = """Problem: Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2, \\\\\nx-5 &\\text{ if } -2 \\le x \\le 2, \\\\\n2x-b &\\text{ if } x <-2.\n\\end{array}\n\\right.\\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper). Solution: For the piecewise function to be continuous, the cases must "meet" at $2$ and $-2$. For example, $ax+3$ and $x-5$ must be equal when $x=2$. This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \\Rightarrow a=-3$. Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$. So $a+b=-3+3=\\boxed{0}$."""
                prompt_txt_ls = [f"{icl_str}\n\nProblem: {line} Solution:" for line in batch_line]

            if api_source == "openai":
                indicater = True
                while indicater:
                    try:
                        responses = ['\t'.join([
                            ele.message.content.replace('\t', '') for ele in completions_with_backoff_openai(
                                client, system_prompt, prompt_txt, model_type, ret_seq
                            ).choices
                        ]) for prompt_txt in prompt_txt_ls]
                        indicater = False
                    except:
                        continue
                
            elif api_source == "google":
                responses = []
                for prompt_txt in prompt_txt_ls:
                    cur_responses = []
                    for _ in range(ret_seq):
                        indicater = True
                        while indicater:
                            try:
                                cur_responses += [completions_with_google(system_prompt, prompt_txt, model_type).replace('\t', '')]
                                indicater = False
                            except:
                                continue
                    responses += ['\t'.join(cur_responses)]
                
            elif api_source == "transformers":
                inputs = tokenizer(prompt_txt_ls, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
                out = model.generate(inputs=inputs.input_ids, max_new_tokens=128)
                responses = tokenizer.batch_decode(out, skip_special_tokens=True)

                if task_type == "mt":
                    if model_type == "mistral_moe":
                        responses = [response.replace(prompt_txt, '').split('\n\n')[0].strip() for response, prompt_txt in zip(responses, prompt_txt_ls)]
                    else:
                        responses = [response.split('### Instruction:')[4].split(f"### {tgt_lang}:")[1].split("\n\n")[0].strip() for response, prompt_txt in zip(responses, prompt_txt_ls)]
                elif task_type == "commonsenseQA" or task_type == "commongen":
                    responses = [response.replace(prompt_txt, '').replace('\n', '\t').strip() for response, prompt_txt in zip(responses, prompt_txt_ls)]
            else:
                print("API source is not supported!")
                exit(1)

            for response in responses:
                out_ls += [response.replace("\n", "") + "\n"]
                pbar.update(1)

    with open(save_name, "w") as f:
        f.writelines(out_ls)
        print(f"{save_name} is saved!")

if __name__ == "__main__":
    main()
