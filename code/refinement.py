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

genai.configure(api_key="AIzaSyD6TPDOsho_SsIGneOHNLjAyN07JCGnwyk")
palm.configure(api_key="AIzaSyD6TPDOsho_SsIGneOHNLjAyN07JCGnwyk")

name_dict = {'vicuna': 'lmsys/vicuna-7b-v1.5', 'llama': 'yahma/llama-7b-hf', 'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',\
             'deepseek': 'deepseek-ai/deepseek-llm-7b-chat', 'deepseek_moe': "deepseek-ai/deepseek-moe-16b-chat", \
             'gpt-neox': 'EleutherAI/gpt-neox-20b', 'gpt-j': "EleutherAI/gpt-j-6b", 'mistral': 'mistralai/Mistral-7B-Instruct-v0.2', \
             'mistral_moe': 'mistralai/Mixtral-8x7B-Instruct-v0.1', "alpaca": "alpaca", "llama2-70b": 'meta-llama/Llama-2-70b-chat-hf', \
             "llama2-13b": 'meta-llama/Llama-2-13b-chat-hf'}

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


@click.command()
@click.option("-lang_dir")
@click.option("-task_type")
@click.option("-start_index", type=int)
@click.option("-api_source", default="google or openai")
@click.option(
    "-model_type", default="model name like gemini, palm2, gpt-3.5-turbo and gpt-4"
)
@click.option("-iteration", type=int)
def main(lang_dir, start_index, iteration, api_source, model_type, task_type):
    if api_source == "openai":
        client = OpenAI()
    elif api_source == "transformers":
        model = AutoModelForCausalLM.from_pretrained(name_dict[model_type], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(name_dict[model_type])
        tokenizer.pad_token = tokenizer.eos_token

    if task_type == "mt":
        src_lines = open(f"srcs/{lang_dir}_src_100.txt", "r").readlines()
    elif task_type == "sci":
        src_lines = []
        for file_name in glob.glob("scibench/dataset/original/*_sol.json"):
            data = json.load(open(file_name))
            src_lines += [ele["problem_text"] for ele in data[1:]]
    else:
        print("Your task type is not supported!")
        exit(1)

    for i in range(start_index, iteration):
        out_name = f"model_outputs/{model_type}/self_refine/{lang_dir}/{model_type}-outputs/{lang_dir}_refinement_100_{model_type}_new_{i}_rerun.txt"
        eval_name = f"model_outputs/{model_type}/self_refine/{lang_dir}/{model_type}-scores/{lang_dir}_eval_100_one-shot_{model_type}_new_{i}_rerun.txt"

        out_lines = open(out_name, "r").readlines()
        eval_lines = open(eval_name, "r").readlines()
        eval_lines = "".join(eval_lines).split("[SEP_TOKEN_WENDA]")[:-1]

        out_ls = []
        eval_ls = []
        with tqdm(total=len(src_lines)) as pbar:
            for src, out, eval in zip(src_lines, out_lines, eval_lines):
                prior_score = (
                    -1 * eval.count("minor")
                    + -5 * eval.count("major")
                    + (-5) * eval.count("critical")
                )
                if task_type == "mt":
                    check_err = "critical" in eval or "major" in eval or "minor" in eval
                elif task_type == "sci":
                    check_err = "False" in eval

                if check_err:
                    if task_type == "mt":
                        prompt_txt = f"""Source: ```{src[:-1]}``` Translation: ```{out[:-1]}``` Feedback: """
                        suffix_prompt = " Improved Yorba-to-English translation:"
                        in_context_txt = f"""Source: ```大众点评乌鲁木齐家居商场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评``` Translation: ```Urumqi Home Furnishing Store Channel provides you with the latest bussiness information such as the address, telephone number, bussiness hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.``` Annotate errors in the translation. MQM annotations: "of high-speed rail" is a critical accuracy/addition error\n"go to the reviews" is a major accuracy/mistranslation error\n"etc.," is a minor style/awkwards error\n\n Source: ```I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.``` Translation: ```Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.``` Annotate errors in the translation. MQM annotations: 'involvement' is a major accuracy/mistranslation error\n'the account holder' is a major accuracy/omission error\n'wäre' is a minor fluency/grammar error\n'dir' is a minor fluency/register error\n\n Source: ```Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.``` Translation: ```Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemže obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.``` Annotate errors in the translation. MQM annotations: 've Vídni' is a major accuracy/addition error\n'the stop-start' is a major accuracy/omission error\n'partaje' is a minor terminology/inappropriate for context error\n\n"""
                        eval_inst_str = f"You are an annotator for the quality of machine translation. Your task is to identify errors and assess the quality of the translation.\nBased on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), locale convention (currency, date, name, telephone, or time format) style (awkward), terminology (inappropriate  for context, inconsistent use), non-translation, other, or no-error.\nEach error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension."
                    elif task_type == "sci":
                        prompt_txt = f"""Question:\n ```{src}```\n Rationale and answer:\n ```{out}```\nFeedback:\n"""
                        suffix_prompt = "Please fix above error and rewrite rationale. New answer (write answer after ####, for example ####0.1):"
                        #eval_prompt_txt = f"""Question: ```{src}``` Answer: ```{out}``` Please evaluate the rationale and answer. Your feedback:"""
                        eval_inst_str = """You are a judge for the rationale of the answer. You will answer in JSON format. Like this, {'correctness': 'True', 'rationale': 'Explanation:'}. If answer is correct, 'correctness' will be 'True', otherwise is 'False'."""
                    if api_source == "openai":
                        # perform refinement based on the feedback
                        response = (
                            client.chat.completions.create(
                                model=model_type,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": prompt_txt + eval + "\n" + suffix_prompt,
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
                        indicater = True
                        while indicater:
                            try:
                                response = completions_with_google(
                                    prompt_txt + eval + "\n" + suffix_prompt,
                                    model_type=model_type,
                                )
                                indicater = False
                            except:
                                continue
                        # perform additional feedback on the refined output
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
                        input_batch = [icl_refinements + [{"role": "user", "content": prompt_txt + eval + "\n" + suffix_prompt}]]
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
                            response = response.split('[/INST]')[4].split("\n")[0].strip()
                        
                        eval_inp_prompt = eval_inst_str + " " + in_context_txt + " " + f"""Source: ```{src}``` Translation: ```{response}``` Annotate errors in the translation. MQM annotations:"""
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

        save_refine_name=f"model_outputs/{model_type}/self_refine/{lang_dir}/{model_type}-outputs/{lang_dir}_refinement_100_{model_type}_new_{i+1}_rerun.txt"
        with open(save_refine_name,"w") as f:
            f.writelines(out_ls)
            print(f"{save_refine_name} is saved!")

        save_eval_name=f"model_outputs/{model_type}/self_refine/{lang_dir}/{model_type}-scores/{lang_dir}_eval_100_one-shot_{model_type}_new_{i+1}_rerun.txt"
        with open(save_eval_name,"w") as f:
            f.writelines(eval_ls)
            print(f"{save_eval_name} is saved!")

if __name__ == "__main__":
    main()
