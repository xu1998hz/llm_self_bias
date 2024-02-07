# OPENAI_API_KEY="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" python3 code/generate_feedback.py -lang_dir yor-en -suffix gpt-3.5-turbo -api_source openai -base_name model_outputs/gpt-3.5-turbo/yor-en_base_outputs_gpt-turbo-3.5.txt -model_type gpt-3.5-turbo
import glob
import json
import click
from openai import OpenAI
from tqdm import tqdm
import google.generativeai as genai
import google.generativeai as palm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoModel
import torch
from datasets import load_dataset
from typing import Dict, TypeVar, Iterable, List

T = TypeVar('T')

genai.configure(api_key="AIzaSyD6TPDOsho_SsIGneOHNLjAyN07JCGnwyk")
palm.configure(api_key="AIzaSyD6TPDOsho_SsIGneOHNLjAyN07JCGnwyk")

name_dict = {'vicuna': 'lmsys/vicuna-7b-v1.5', 'llama': 'yahma/llama-7b-hf', 'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',\
             'deepseek': 'deepseek-ai/deepseek-llm-7b-chat', 'deepseek_moe': "deepseek-ai/deepseek-moe-16b-chat", 'mistral-base': 'mistralai/Mistral-7B-v0.1', \
             'gpt-neox': 'EleutherAI/gpt-neox-20b', 'gpt-j': "EleutherAI/gpt-j-6b", 'mistral-inst2': 'mistralai/Mistral-7B-Instruct-v0.2', \
             'mistral_moe': 'mistralai/Mixtral-8x7B-Instruct-v0.1', "alpaca": "alpaca", "llama2-70b": 'meta-llama/Llama-2-70b-chat-hf', \
             "llama2-13b": 'meta-llama/Llama-2-13b-chat-hf', "ift": "/mnt/taurus/home/guangleizhu/reproduce_pinpoint/finetune/ft_out/mistral_ft_test/checkpoint-66",\
             "eft": "/home/guangleizhu/peril_self_improve/instruct_ft/ckpt/mistral_eft/checkpoint-156", 'mistral-inst1': 'mistralai/Mistral-7B-Instruct-v0.1', \
             'mistral-inst1-mqm': "instruct_ft/ckpt/mistral_instruct_mqm_ift/checkpoint-187/", "mistral-inst2-mqm": "/home/guangleizhu/peril_self_improve/instruct_ft/ckpt/mistral_inst_ift_mqm/checkpoint-187",\
             'mistral-inst1-mqm_fixed': "instruct_ft/ckpt/mistral_instruct_mqm_ift_fixed/checkpoint-187"}

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

def completions_with_google(prompt_txt, inst_str, model_type):
    if model_type == "gemini":
        model = genai.GenerativeModel(model_name="gemini-pro")
        completion = model.generate_content(
            inst_str + " " + prompt_txt,
            generation_config={"temperature": 1.0, "max_output_tokens": 1024},
        )
        try:
            return completion.text
        except:
            return "[BLOCKED]"
    elif model_type == "palm2":
        completion = palm.generate_text(
            model="models/text-bison-001",
            prompt=inst_str + " " + prompt_txt,
            temperature=1.0,
            max_output_tokens=1024,
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
@click.option("-lang_dir", help="zh-en")
@click.option("-savename")
@click.option("-api_source", help="google or openai")
@click.option("-base_name", help="zh-en_base_outputs_500.txt")
@click.option(
    "-model_type", help="model name like gemini, palm2, gpt-3.5-turbo and gpt-4"
)
@click.option("-last_feedback", help="last feedback file", default=None)
@click.option("-batch_size", type=int)
def main(lang_dir, savename, base_name, api_source, model_type, task_type, last_feedback, batch_size):
    if api_source == "openai":
        client = OpenAI()
    elif api_source == "transformers":
        model = AutoModelForCausalLM.from_pretrained(name_dict[model_type], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(name_dict[model_type])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "[PAD]"
            tokenizer.padding_side = "left"
            print(f"Padding token is not found, setting padding token to [PAD]")

    if task_type == "mt":
        instruction_str = f"You are an annotator for the quality of machine translation. Your task is to identify errors and assess the quality of the translation.\nBased on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), locale convention (currency, date, name, telephone, or time format) style (awkward), terminology (inappropriate  for context, inconsistent use), non-translation, other, or no-error.\nEach error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension."
        in_context_txt = f"""Source: ```大众点评乌鲁木齐家居商场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评``` Translation: ```Urumqi Home Furnishing Store Channel provides you with the latest bussiness information such as the address, telephone number, bussiness hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.``` Annotate errors in the translation. MQM annotations: "of high-speed rail" is a critical accuracy/addition error\n"go to the reviews" is a major accuracy/mistranslation error\n"etc.," is a minor style/awkwards error\n\n Source: ```I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.``` Translation: ```Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.``` Annotate errors in the translation. MQM annotations: 'involvement' is a major accuracy/mistranslation error\n'the account holder' is a major accuracy/omission error\n'wäre' is a minor fluency/grammar error\n'dir' is a minor fluency/register error\n\n Source: ```Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.``` Translation: ```Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemže obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.``` Annotate errors in the translation. MQM annotations: 've Vídni' is a major accuracy/addition error\n'the stop-start' is a major accuracy/omission error\n'partaje' is a minor terminology/inappropriate for context error\n\n"""
    elif task_type == "commonsenseQA":
        in_context_txt = f"""Q: A fencing thrust with a sharp sword towards a person would result in what?\n\nAnswer Choices: A) injury, B) small cuts, C) fever, D) competition, E) puncture wound\n\nA: In a controlled fencing match with a sharp sword, a fencing thrust is likely to result in (D) competition rather than injury or a puncture wound. Therefore, the correct final answer is (D) competition\n\n(answer: D)\n\nReview this answer and tell me if it is the correct or incorrect answer.\n\nThe previous answer is incorrect. A fencing thrust with a sharp sword towards a person would result in (A) injury or (E) puncture wound. Therefore the correct answer is either (A) or (E), not (D) competition. The reference to competition presumably refers to the context in which this action might occur, rather than the direct result of the action itself.\n\nIncorrect\n\n""" # Q: A fencing thrust with a sharp sword towards a person would result in what?\n\nAnswer Choices: A) injury, B) small cuts, C) fever, D) competition, E) puncture wound\n\nA: Sammy would likely go to populated areas if he wants to be where the people are. Although there may be people in areas like a race track or an apartment, these are specific places that don't always guarantee the presence of people. Populated areas, on the other hand, are generally guaranteed to have people. The desert and a roadblock are also less likely areas for people to gather. So, the best answer is B) populated areas.\n\n(answer: B)\n\nReview this answer and tell me if it is the correct or incorrect answer.\n\nThe previous answer is correct\n\nCorrect\n\n
    else:
        print("Task is not supported!")
        exit(1)

    if task_type == "mt":
        src_lines = open(f"srcs/{lang_dir}_src_100.txt", "r").readlines()
        out_lines = open(base_name, "r").readlines()
    elif task_type == "commonsenseQA":
        src_lines = [{'question': ele['question'], 'choices': ele['choices']} for ele in load_dataset('tau/commonsense_qa')['test']][:200]
        out_lines = open(base_name, "r").readlines()
    else:
        print(f"{task_type} is not supported!")
        exit(1)

    if last_feedback:
        eval_lines = open(last_feedback, "r").readlines()
        eval_lines = "".join(eval_lines).split("[SEP_TOKEN_WENDA]")[:-1]

    out_ls = []
    with tqdm(total=len(src_lines)) as pbar:
        for index, (src_batch_txt, src_batch_out) in enumerate(zip(batchify(src_lines, batch_size), batchify(out_lines, batch_size))):
            if task_type == "mt":
                prompt_txt_ls = [(
                    in_context_txt
                    + f""" Source: ```{src_txt[:-1]}``` Translation: ```{out[:-1]}``` Annotate errors in the translation. MQM annotations:"""
                ) for src_txt, out in zip(src_batch_txt, src_batch_out)]
            elif task_type == "commonsenseQA":
                new_out_ls = [out[:-1].replace('\t\t', '\n\n') for out in src_batch_out]
                prompt_txt_ls = [(
                    in_context_txt
                    + 
                    f"""Q: {src_txt['question']}\n\nAnswer Choices: A) {src_txt['choices']['text'][0]}, B) {src_txt['choices']['text'][1]}, C) {src_txt['choices']['text'][2]}, D) {src_txt['choices']['text'][3]}, E) {src_txt['choices']['text'][4]}\n\nA: {new_out}\n\nReview this answer and tell me if it is the correct or incorrect answer.\n\n"""
                ) for src_txt, new_out in zip(src_batch_txt, new_out_ls)]
            else:
                print(f"{task_type} is not supported!")
                exit(1)

            # if last_feedback:
            #     eval = eval_lines[index]
            #     if task_type == "mt":
            #         check_err = "critical" in eval or "major" in eval or "minor" in eval
            #     elif task_type == "sci":
            #         check_err = "False" in eval

            if api_source == "openai":
                # if (not last_feedback) or (last_feedback and check_err):
                response_ls = [(
                    client.chat.completions.create(
                        model=model_type,
                        messages=[
                            {
                                "role": "system",
                                "content": instruction_str,
                            },
                            {"role": "user", "content": prompt_txt},
                        ],
                        temperature=1.0,
                        max_tokens=1024,
                        top_p=1,
                    )
                    .choices[0]
                    .message.content
                ) for prompt_txt in prompt_txt_ls]
                # else:
                #     response = eval_lines[index]
            elif api_source == "google":
                # if (not last_feedback) or (last_feedback and check_err):
                indicater = True
                while indicater:
                    try:
                        response_ls = [completions_with_google(
                            prompt_txt,
                            instruction_str,
                            model_type=model_type,
                        ) for prompt_txt in prompt_txt_ls]
                        indicater = False
                    except:
                        continue
                # else:
                #     response = eval_lines[index]
            
            elif api_source == "transformers":
                inputs = tokenizer([instruction_str + " " +  prompt_txt for prompt_txt in prompt_txt_ls], return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
                out = model.generate(inputs=inputs.input_ids, max_new_tokens=128)
                response_ls = tokenizer.batch_decode(out, skip_special_tokens=True)
                if task_type == "mt":
                    if model_type == "mistral_moe":
                        response_ls = [response.replace(instruction_str + " " +  prompt_txt, "").split("\n\n")[0].strip() for prompt_txt, response in zip(prompt_txt_ls, response_ls)]
                    else:
                        response_ls = [response.split("Annotate errors in the translation. MQM annotations:")[4].split("\n\n")[0].strip() for prompt_txt, response in zip(prompt_txt_ls, response_ls)]
                else:
                    response_ls = [response.replace(prompt_txt, "").replace("\n", '\t').strip() for prompt_txt, response in zip(prompt_txt_ls, response_ls)]
            else:
                print("API source is not found!")
                exit(1)

            for response in response_ls:
                out_ls += [response + "[SEP_TOKEN_WENDA]"]
                pbar.update(1)

    with open(savename,"w") as f:
        f.writelines(out_ls)
        print(f"{savename} is saved!")

if __name__ == "__main__":
    main()
