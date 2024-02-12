# OPENAI_API_KEY="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" python3 code/refinement.py -lang_dir zh-en -out zh-en_self_refinement_500_1.txt -eval zh-en_self_eval_500_one-shot_src_refine_2.txt -suffix 2
import click
from openai import OpenAI
from tqdm import tqdm
import google.generativeai as genai
import google.generativeai as palm
import glob
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoModel
import torch
from datasets import load_dataset
from typing import Dict, TypeVar, Iterable, List
from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers

T = TypeVar('T')

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )

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

def completions_with_google(prompt_txt, model_type):
    if model_type == "gemini":
        model = genai.GenerativeModel(model_name="gemini-pro")
        completion = model.generate_content(
            prompt_txt,
            generation_config={"temperature": 1.0, "max_output_tokens": 1024},
        )
        try:
            return completion.text
        except:
            return "[BLOCKED]"
    elif model_type == "palm2":
        completion = palm.generate_text(
            model="models/text-bison-001",
            prompt=prompt_txt,
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
@click.option("-lang_dir")
@click.option("-task_type")
@click.option("-start_index", type=int)
@click.option("-api_source", help="google or openai")
@click.option(
    "-model_type", help="model name like gemini, palm2, gpt-3.5-turbo and gpt-4"
)
@click.option(
    "-instructscore_enable", help="True or False"
)
@click.option("-iteration", type=int)
def main(lang_dir, start_index, iteration, api_source, model_type, task_type, instructscore_enable):
    if api_source == "openai":
        client = OpenAI()
    elif api_source == "transformers":
        model = AutoModelForCausalLM.from_pretrained(name_dict[model_type], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(name_dict[model_type])
        tokenizer.pad_token = tokenizer.eos_token

    # load instructscore if enabled
    if instructscore_enable:
        inst_tokenizer = LlamaTokenizer.from_pretrained(
            "xu1998hz/InstructScore", model_max_length=512, use_fast=False
        )
        
        inst_model = LlamaForCausalLM.from_pretrained("xu1998hz/InstructScore", torch_dtype=torch.bfloat16, device_map="auto")
        inst_tokenizer.padding_side = "left"
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=inst_tokenizer,
        )
        inst_model.eval()
        ref_ls = open('refs/en_ref_100.txt', 'r').readlines()

    if task_type == "mt":
        src_lines = open(f"srcs/{lang_dir}_src_100.txt", "r").readlines()
        icl_refinements = [
            {
                "role": "user",
                "content": """Please fix all errors. You can rewrite translation if translation is bad. Source: ```大众点评乌鲁木齐家居商场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评``` Translation: ```Urumqi Home Furnishing Store Channel provides you with the latest bussiness information such as the address, telephone number, bussiness hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.``` Feedback: 'of high-speed rail' is a critical accuracy/addition error\n'go to the reviews' is a major accuracy/mistranslation error\n'etc.,' is a minor style/awkward error\n Improved Chinese-to-English translation:""",
            },
            {
                "role": "assistant",
                "content": """Dianping Urumqi Renovation and Design Channel will provide you with the address, phone number, operation time and other information of HSR Easyhome, and please come to Dianping if you are looking for a renovation company.\n\n""",
            },
            {
                "role": "user",
                "content": "Source: ```I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.``` Translation: ```Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.``` Feedback: 'involvement' is a major accuracy/mistranslation error\n'the account holder' is a major accuracy/omission error\n'wäre' is a minor fluency/grammar error\n'dir' is a minor fluency/register error\n Improved English-to-German translation:",
            },
            {
                "role": "assistant",
                "content": """Ich bitte um Entschuldigung, aber wir benötigen das Einverständnis des Kontoinhabers, um eine Bestellung mit einer anderen Person zu besprechen, falls es schon eingeholt wurde, entschuldige ich mich, aber ich kann dies ohne das Einverständnis des Kontoinhabers nicht mit Ihnen besprechen.\n\n""",
            },
            {
                "role": "user",
                "content": "Source: ```Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.``` Translation: ```Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemže obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.``` Feedback: 've Vídni' is a major accuracy/addition error\n'the stop-start' is a major accuracy/omission error\n'partaje' is a minor terminology/inappropriate for context error\n Improved English-to-Cezch translation:",
            },
            {
                "role": "assistant",
                "content": """Ve Vídni byly obnoveny rozhovory o oživení jaderného paktu a obě strany se snaží odhadnout, jaké jsou vyhlídky na úspěch po posledních výměnách názorů v rámci přerušených jednání.\n\n""",
            },
        ]
    elif task_type == "commonsenseQA":
        icl_refinements = [
            {
                "role": "user",
                "content": """Q: A fencing thrust with a sharp sword towards a person would result in what?\n\nAnswer Choices: A) injury, B) small cuts, C) fever, D) competition, E) puncture wound\n\nA: In a controlled fencing match with a sharp sword, a fencing thrust is likely to result in (D) competition rather than injury or a puncture wound. Therefore, the correct final answer is (D) competition\n\n(answer: D)\n\nFeedback: The previous answer is incorrect. A fencing thrust with a sharp sword towards a person would result in (A) injury or (E) puncture wound. Therefore the correct answer is either (A) or (E), not (D) competition. The reference to competition presumably refers to the context in which this action might occur, rather than the direct result of the action itself.\n\nNew answer:""",
            },
            {
                "role": "assistant",
                "content": """A fencing thrust with a sharp sword towards a person would result in a potential injury or puncture wound. So, the corrected answer is an injury as the injury is a broad term that can include a puncture wound. Therefore, the correct final answer is (A) injury.\n\n(answer: A)""",
            },
        ]
        src_lines = [{'question': ele['question'], 'choices': ele['choices']} for ele in load_dataset('tau/commonsense_qa')['test']][:200]
    else:
        print("Your task type is not supported!")
        exit(1)

    for i in range(start_index, iteration):
        out_name = f"model_outputs-inst/{model_type}/self_refine/{lang_dir}/{model_type}-outputs/{lang_dir}_refinement_100_{model_type}_new_{i}_rerun.txt"
        eval_name = f"model_outputs-inst/{model_type}/self_refine/{lang_dir}/{model_type}-scores/{lang_dir}_eval_100_one-shot_{model_type}_new_{i}_rerun.txt"

        out_lines = open(out_name, "r").readlines()
        eval_lines = open(eval_name, "r").readlines()
        eval_lines = "".join(eval_lines).split("[SEP_TOKEN_WENDA]")[:-1]

        out_ls = []
        eval_ls = []
        with tqdm(total=len(src_lines)) as pbar:
            for index, (src, out, eval) in enumerate(zip(src_lines, out_lines, eval_lines)):
                if instructscore_enable:
                    prior_score = (
                        -1 * eval.count("Major/minor: Minor")
                        + -5 * eval.count("Major/minor: Major")
                    )
                else:
                    prior_score = (
                        -1 * eval.count("minor")
                        + -5 * eval.count("major")
                        + (-5) * eval.count("critical")
                    )

                if task_type == "mt":
                    if instructscore_enable:
                        check_err = "major" in eval.lower() or "minor" in eval.lower()
                    else:
                        check_err = "critical" in eval or "major" in eval or "minor" in eval
                elif task_type == "commonsenseQA":
                    check_err = "incorrect" in eval.split('\t')[1].lower()

                if check_err:
                    if task_type == "mt":
                        prompt_txt = f"""Source: ```{src[:-1]}``` Translation: ```{out[:-1]}``` Feedback: """
                        suffix_prompt = " Improved Yorba-to-English translation:"
                        in_context_txt = f"""Source: ```大众点评乌鲁木齐家居商场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评``` Translation: ```Urumqi Home Furnishing Store Channel provides you with the latest bussiness information such as the address, telephone number, bussiness hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.``` Annotate errors in the translation. MQM annotations: "of high-speed rail" is a critical accuracy/addition error\n"go to the reviews" is a major accuracy/mistranslation error\n"etc.," is a minor style/awkwards error\n\n Source: ```I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.``` Translation: ```Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.``` Annotate errors in the translation. MQM annotations: 'involvement' is a major accuracy/mistranslation error\n'the account holder' is a major accuracy/omission error\n'wäre' is a minor fluency/grammar error\n'dir' is a minor fluency/register error\n\n Source: ```Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.``` Translation: ```Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemže obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.``` Annotate errors in the translation. MQM annotations: 've Vídni' is a major accuracy/addition error\n'the stop-start' is a major accuracy/omission error\n'partaje' is a minor terminology/inappropriate for context error\n\n"""
                        eval_inst_str = f"You are an annotator for the quality of machine translation. Your task is to identify errors and assess the quality of the translation.\nBased on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), locale convention (currency, date, name, telephone, or time format) style (awkward), terminology (inappropriate  for context, inconsistent use), non-translation, other, or no-error.\nEach error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension."
                        prompt_txt = prompt_txt + eval.split("[PAD]")[0].strip() + "\n" + suffix_prompt
                    elif task_type == "commonsenseQA":
                        new_out = out[:-1].replace('\t\t', '\n\n')
                        new_eval = eval.split('\t\t')[0]
                        prompt_txt = f"""Q: {src['question']}\n\nAnswer Choices: Choices: A) {line['choices']['text'][0]}, B) {line['choices']['text'][1]}, C) {line['choices']['text'][2]}, D) {line['choices']['text'][3]}, E) {line['choices']['text'][4]}\n\nA: {new_out}\n\nFeedback: {new_eval}\n\nNew answer:"""
                    if api_source == "openai":
                        # perform refinement based on the feedback
                        response = (
                            client.chat.completions.create(
                                model=model_type,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": prompt_txt,
                                    },
                                ],
                                temperature=1.0,
                                max_tokens=1024,
                                top_p=1,
                            )
                            .choices[0]
                            .message.content
                        )
                        # perform additional feedback on the refined output
                        if instructscore_enable:
                            eval_prompt_txt=f'You are evaluating Chinese-to-English Machine translation task. The correct translation is "{ref_ls[index][:-1]}". The model generated translation is "{response}". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error type, major/minor label, error location of the model generated translation and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don\'t lead to loss of meaning but will be noticed.'
                            eval_inputs = inst_tokenizer(eval_prompt_txt, return_tensors="pt").to(inst_model.device)
                            eval_output = inst_model.generate(inputs=eval_inputs.input_ids, max_new_tokens=512)
                            eval_response = inst_tokenizer.batch_decode(eval_output, skip_special_tokens=True)[0]
                        else:
                            eval_response = (
                                client.chat.completions.create(
                                    model=model_type,
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": eval_inst_str,
                                        },
                                        {"role": "user", "content": in_context_txt + " " + f"""Source: ```{src}``` Translation: ```{response}``` MQM annotations:"""},
                                    ],
                                    temperature=1.0,
                                    max_tokens=1024,
                                    top_p=1,
                                )
                                .choices[0]
                                .message.content
                            )
                    elif api_source == "google":
                        # perform refinement based on the feedback
                        # print(prompt_txt)
                        indicater = True
                        while indicater:
                            try:
                                response = completions_with_google(
                                    prompt_txt,
                                    model_type=model_type,
                                )
                                indicater = False
                            except:
                                continue
                        # perform additional feedback on the refined output
                        if instructscore_enable:
                            eval_prompt_txt=[f'You are evaluating Chinese-to-English Machine translation task. The correct translation is "{ref_ls[index][:-1]}". The model generated translation is "{response}". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error type, major/minor label, error location of the model generated translation and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don\'t lead to loss of meaning but will be noticed.']
                            eval_inputs = inst_tokenizer(eval_prompt_txt, return_tensors="pt").to(inst_model.device)
                            eval_output = inst_model.generate(inputs=eval_inputs.input_ids, max_new_tokens=512)
                            eval_response = inst_tokenizer.batch_decode(eval_output, skip_special_tokens=True)[0].replace(eval_prompt_txt[0],'')
                            # print(eval_prompt_txt)
                            # print('-'*50)
                            # print(eval_response)
                            # print('-'*100)
                        else:
                            indicater = True
                            while indicater:
                                try:
                                    eval_response = completions_with_google(
                                        eval_inst_str + " " + in_context_txt + " " + f"""Source: ```{src}``` Translation: ```{response}``` MQM annotations:""",
                                        model_type=model_type,
                                    )
                                    indicater = False
                                except:
                                    continue
                    elif api_source == "transformers":
                        model.generation_config = GenerationConfig.from_pretrained(name_dict[model_type])
                        model.generation_config.pad_token_id = model.generation_config.eos_token_id
                        input_batch = [icl_refinements + [{"role": "user", "content": prompt_txt}]]
                        input_batch = [tokenizer.apply_chat_template(ele, add_generation_prompt=True, tokenize=False) for ele in input_batch] 
                        inputs = tokenizer(input_batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
                        output = model.generate(inputs=inputs.input_ids, max_new_tokens=128)
                        response = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                        
                        if model_type[:6] == "llama2":
                            if 'improved translation' in response.split('[/INST]')[4].split("\n")[0].lower():
                                response = response.split('[/INST]')[4].split("\n")[2]
                            else:
                                response = response.split('[/INST]')[4].split("\n")[0].strip()
                        else:
                            if model_type == "deepseek_moe":
                                response = response.split('Assistant:')[4].split("\n")[0].strip()
                            else:
                                response = response.split('[/INST]')[4].split("\n")[0].strip()
                        
                        if task_type == "mt":
                            eval_inp_prompt = eval_inst_str + " " + in_context_txt + " " + f"""Source: ```{src}``` Translation: ```{response}``` Annotate errors in the translation. MQM annotations:"""
                        else:
                            # If I have other tasks and I will modiffy them for here
                            pass
                        eval_inputs = tokenizer([eval_inp_prompt], return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
                        eval_out = model.generate(inputs=eval_inputs.input_ids, max_new_tokens=128)
                        eval_response = tokenizer.batch_decode(eval_out, skip_special_tokens=True)[0]
                        if model_type == "mistral_moe":
                            eval_response = eval_response.replace(eval_inp_prompt, "").split("\n\n")[0]
                        else:
                            eval_response = eval_response.split("MQM annotations:")[4].split("\n\n")[0].strip()

                    else:
                        print("API source is not found!")
                        exit(1)
                    
                    if instructscore_enable:
                        post_score = (
                            -1 * eval_response.count("Major/minor: Minor")
                            + -5 * eval_response.count("Major/minor: Major")
                        )
                    else:
                        post_score = (
                            -1 * eval_response.count("minor")
                            + -5 * eval_response.count("major")
                            + (-5) * eval_response.count("critical")
                        )
                    if post_score > prior_score:
                        out_ls += [response.replace("\n", "").replace("\t", "") + "\n"]
                        eval_ls += [eval_response + "[SEP_TOKEN_WENDA]"]
                    else:
                        out_ls += [out]
                        eval_ls += [eval + "[SEP_TOKEN_WENDA]"]
                else:
                    out_ls += [out]
                    eval_ls += [eval + "[SEP_TOKEN_WENDA]"]
                pbar.update(1)

        if task_type == "mt":
            save_refine_name=f"model_outputs-inst/{model_type}/self_refine/{lang_dir}/{model_type}-outputs/{lang_dir}_refinement_100_{model_type}_new_{i+1}_rerun.txt"
            save_eval_name=f"model_outputs-inst/{model_type}/self_refine/{lang_dir}/{model_type}-scores/{lang_dir}_eval_100_one-shot_{model_type}_new_{i+1}_rerun.txt"
        else:
            save_refine_name=f"model_outputs-inst/{model_type}/self_refine/{task_type}/{model_type}-outputs/{task_type}_refinement_100_{model_type}_new_{i+1}_rerun.txt"
            save_eval_name=f"model_outputs-inst/{model_type}/self_refine/{task_type}/{model_type}-scores/{task_type}_eval_100_one-shot_{model_type}_new_{i+1}_rerun.txt"

        with open(save_refine_name,"w") as f:
            f.writelines(out_ls)
            print(f"{save_refine_name} is saved!")

        with open(save_eval_name,"w") as f:
            f.writelines(eval_ls)
            print(f"{save_eval_name} is saved!")

if __name__ == "__main__":
    main()
