# OPENAI_API_KEY="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" python3 code/refinement.py -lang_dir zh-en -out zh-en_self_refinement_500_1.txt -eval zh-en_self_eval_500_one-shot_src_refine_2.txt -suffix 2
import click
from openai import OpenAI
from tqdm import tqdm
import google.generativeai as genai
import google.generativeai as palm
import glob
import json

genai.configure(api_key="AIzaSyD6TPDOsho_SsIGneOHNLjAyN07JCGnwyk")
palm.configure(api_key="AIzaSyD6TPDOsho_SsIGneOHNLjAyN07JCGnwyk")


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
@click.option("-out", help="zh-en_self_refinement_500_1.txt")
@click.option("-eval", help="zh-en_self_eval_500_one-shot_src_refine_2.txt")
@click.option("-suffix")
@click.option("-eval_suffix")
@click.option("-api_source", default="google or openai")
@click.option(
    "-model_type", default="model name like gemini, palm2, gpt-3.5-turbo and gpt-4"
)
def main(lang_dir, out, eval, suffix, api_source, model_type, task_type, eval_suffix):
    if api_source == "openai":
        client = OpenAI()

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

    out_lines = open(out, "r").readlines()
    eval_lines = open(eval, "r").readlines()
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
                    prompt_txt = f"""Source:\n ```{src}```\n Translation:\n ```{out}```\nFeedback:\n"""
                    suffix_prompt = "Please fix above error. You can rewrite translation if translation is bad. Improved translation:"
                    in_context_txt = f"""Source: ```大众点评乌鲁木齐家居商场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评``` Translation: ```Urumqi Home Furnishing Store Channel provides you with the latest bussiness information such as the address, telephone number, bussiness hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.``` MQM annotations: "of high-speed rail" is a critical accuracy/addition error\n"go to the reviews" is a major accuracy/mistranslation error\n"etc.," is a minor style/awkwards error\n\n Source: ```I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.``` Translation: ```Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.``` MQM annotations: 'involvement' is a major accuracy/mistranslation error\n'the account holder' is a major accuracy/omission error\n'wäre' is a minor fluency/grammar error\n'dir' is a minor fluency/register error\n\n Source: ```Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.``` Translation: ```Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemže obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.``` MQM annotations: 've Vídni' is a major accuracy/addition error\n'the stop-start' is a major accuracy/omission error\n'partaje' is a minor terminology/inappropriate for context error\n\n"""
                    eval_prompt_txt = (
                        in_context_txt
                        + f"""Source: ```{src}``` Translation: ```{out}``` MQM annotations:"""
                    )
                    eval_inst_str = f"You are an annotator for the quality of machine translation. Your task is to identify errors and assess the quality of the translation.\nBased on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), locale convention (currency, date, name, telephone, or time format) style (awkward), terminology (inappropriate  for context, inconsistent use), non-translation, other, or no-error.\nEach error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension."
                elif task_type == "sci":
                    prompt_txt = f"""Question:\n ```{src}```\n Rationale and answer:\n ```{out}```\nFeedback:\n"""
                    suffix_prompt = "Please fix above error and rewrite rationale. New answer (write answer after ####, for example ####0.1):"
                    eval_prompt_txt = f"""Question: ```{src}``` Answer: ```{out}``` Please evaluate the rationale and answer. Your feedback:"""
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
                                {"role": "user", "content": eval_prompt_txt},
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
                                eval_inst_str + " " + eval_prompt_txt,
                                model_type=model_type,
                            )
                            indicater = False
                        except:
                            continue
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

    if task_type == "mt":
        with open(
            f"model_outputs/{model_type}/{lang_dir}_self_refinement_100_{suffix}.txt",
            "w",
        ) as f:
            f.writelines(out_ls)
            print(
                f"model_outputs/{model_type}/{lang_dir}_self_refinement_100_{suffix}.txt is saved!"
            )

        with open(
            f"model_outputs/{model_type}/{lang_dir}_eval_100_one-shot_{eval_suffix}.txt",
            "w",
        ) as f:
            f.writelines(eval_ls)
            print(
                f"model_outputs/{model_type}/{lang_dir}_eval_100_one-shot_{eval_suffix}.txt is saved!"
            )
    else:
        with open(
            f"model_outputs/{model_type}/{task_type}_self_refinement_100_{suffix}.txt",
            "w",
        ) as f:
            f.writelines(out_ls)
            print(
                f"model_outputs/{model_type}/{task_type}_self_refinement_100_{suffix}.txt is saved!"
            )

        with open(
            f"model_outputs/{model_type}/{task_type}_eval_100_one-shot_{eval_suffix}.txt",
            "w",
        ) as f:
            f.writelines(eval_ls)
            print(
                f"model_outputs/{model_type}/{task_type}_eval_100_one-shot_{eval_suffix}.txt is saved!"
            )


if __name__ == "__main__":
    main()
