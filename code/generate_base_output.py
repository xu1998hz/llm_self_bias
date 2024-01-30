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

genai.configure(api_key="AIzaSyD6TPDOsho_SsIGneOHNLjAyN07JCGnwyk")
palm.configure(api_key="AIzaSyD6TPDOsho_SsIGneOHNLjAyN07JCGnwyk")


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
def main(lang_dir, api_source, model_type, task_type):
    if api_source == "openai":
        client = OpenAI()

    if task_type == "mt":
        system_prompt = f"You are translating {src_lang}-to-{tgt_lang} machine translation. Do not provide any explanations or text apart from the translation. "
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
            response = completions_with_google(system_prompt, prompt_txt, model_type)
        else:
            print("API source is not supported!")
            exit(1)

        out_ls += [response.replace("\n", "\t") + "\n"]

    if task_type == "mt":
        with open(
            f"model_outputs/{model_type}/{lang_dir}_base_outputs_{model_type}.txt", "w"
        ) as f:
            f.writelines(out_ls)
            print(
                f"model_outputs/{model_type}/{lang_dir}_base_outputs_{model_type}.txt is saved!"
            )
    else:
        with open(
            f"model_outputs/{model_type}/{task_type}_base_outputs_{model_type}.txt", "w"
        ) as f:
            f.writelines(out_ls)
            print(
                f"model_outputs/{model_type}/{task_type}_base_outputs_{model_type}.txt is saved!"
            )


if __name__ == "__main__":
    main()