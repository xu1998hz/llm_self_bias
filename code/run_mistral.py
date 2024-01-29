from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import click
from tqdm import tqdm


def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : min(i + batch_size, len(lst))]


device_id = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.padding_side = "left"
# load in model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2", cache_dir="/mnt/data4/wendaxu"
).to(device_id)
model.resize_token_embeddings(len(tokenizer))
model.eval()

icl_trans = {
    "zh-en": [
        {"role": "user", "content": "Chinese: '新华时评：把优秀返乡农民工打造成乡村振兴生力军-新华网' English: "},
        {
            "role": "assistant",
            "content": "Xinhua Commentary: Outstanding returning rural migrant workers can be a rural revitalization army - Xinhuanet\n",
        },
        {"role": "user", "content": "Chinese: '乡村振兴，人才是关键。' English: "},
        {
            "role": "assistant",
            "content": "Talent is the key to rural revitalization.\n",
        },
        {"role": "user", "content": "Chinese: '人才哪里找？' English: "},
        {
            "role": "assistant",
            "content": "Where is talent found?\n",
        },
    ],
    "en-de": [
        {
            "role": "user",
            "content": "English: 'Couple MACED at California dog park for not wearing face masks while having lunch (VIDEO) - RT USA News' German: ",
        },
        {
            "role": "assistant",
            "content": "Paar in Hundepark in Kalifornien mit Pfefferspray besprüht, weil es beim Mittagessen keine Masken trug (VIDEO) - RT USA News\n",
        },
        {
            "role": "user",
            "content": "English: 'There's mask-shaming and then there's full on assault.' German: ",
        },
        {
            "role": "assistant",
            "content": "Masken-Shaming ist eine Sache, Körperverletzung eine andere.\n",
        },
        {
            "role": "user",
            "content": "English: 'A California couple having lunch with their puppy at a dog park experienced the second, when an irate woman pepper-sprayed them for refusing to mask up... while eating.' German: ",
        },
        {
            "role": "assistant",
            "content": """Ein kalifornisches Paar, das mit seinem Welpen in einem Hundepark zu Mittag aß, erlebte Letztere, als eine wütende Frau sie mit Pfefferspray besprühte, weil sie keine Masken tragen wollten – während sie aßen.\n""",
        },
    ],
    "yor-en": [
        {
            "role": "user",
            "content": "Yoruba: 'Àwọn onímọ̀ ìjìnlẹ̀ sáyẹ̀nsì láti ilé ìkẹ́ẹ̀kọ́ gíga ti ìsègun Stanford lọ́jọ́ ajé ti kéde ìdásílẹ̀ irinṣẹ́ ìwádìí tuntun tí ó le tó nǹkan lẹ́sẹẹsẹ pẹ̀lú bí wọ́n bá se rí: irinṣẹ́ kéreké tí a lè tẹ̀ jáde pẹ̀lú lílo irinṣẹ́ ìtẹ̀wé ìgbàlódé pẹ̀lú owó ẹyọ fún ọ̀kọ̀ọ̀kan.' English: ",
        },
        {
            "role": "assistant",
            "content": "On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each.\n",
        },
        {
            "role": "user",
            "content": "Yoruba: 'Olùwádìí àgbà sọ pé èyí le è tètè mú kí wọn tètè mọ nípa jẹjẹrẹ, ikọ́ fee HIV, àti ibà lára àwọn aláàrẹ̀ ni àwọn orílẹ̀ èdè tí kòní owó púpọ̀, ní ibi tí àwọn tí wọ́n ń borí àìsàn bíì jẹjẹrẹ ọyàn jẹ bí ìlàjì àwọn olówó ní ìlú náà.' English: ",
        },
        {
            "role": "assistant",
            "content": "Lead researchers say this may bring early detection of cancer, tuberculosis, HIV and malaria to patients in low-income countries, where the survival rates for illnesses such as breast cancer can be half those of richer countries.\n",
        },
        {
            "role": "user",
            "content": "Yoruba: 'Jas 39C lọ kọlu ibi tí ènìyàn ma ń gbà ní ago mẹsán àbọ̀ (0230 UTC) ó sì jọ́nà, tó sì ti àwọn ọkọ̀ òfuruf;u pa.' English: ",
        },
        {
            "role": "assistant",
            "content": "The JAS 39C Gripen crashed onto a runway at around 9:30 am local time (0230 UTC) and exploded, closing the airport to commercial flights.\n",
        },
    ],
}


icl_eval = [
    {
        "role": "user",
        "content": """You are an annotator for the quality of machine translation. Your task is to identify errors and assess the quality of the translation. Based on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), locale convention (currency, date, name, telephone, or time format) style (awkward), terminology (inappropriate  for context, inconsistent use), non-translation, other, or no-error.\nEach error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension. Source: ```大众点评乌鲁木齐家居商场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评``` Translation: ```Urumqi Home Furnishing Store Channel provides you with the latest bussiness information such as the address, telephone number, bussiness hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.``` MQM annotations:""",
    },
    {
        "role": "assistant",
        "content": """'of high-speed rail' is a critical accuracy/addition error\n'go to the reviews' is a major accuracy/mistranslation error\n'etc.,' is a minor style/awkward error\n\n""",
    },
    {
        "role": "user",
        "content": "Source: ```I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.``` Translation: ```Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.``` MQM annotations:",
    },
    {
        "role": "assistant",
        "content": """'involvement' is a major accuracy/mistranslation error\n'the account holder' is a major accuracy/omission error\n'wäre' is a minor fluency/grammar error\n'dir' is a minor fluency/register error\n\n""",
    },
    {
        "role": "user",
        "content": "Source: ```Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.``` Translation: ```Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemže obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.``` MQM annotations:",
    },
    {
        "role": "assistant",
        "content": """'ve Vídni' is a major accuracy/addition error\n'the stop-start' is a major accuracy/omission error\n'partaje' is a minor terminology/inappropriate for context error\n\n""",
    },
]

ignore_outputs = [
    """'of high-speed rail' is a """,  # critical accuracy/addition error""",
    """'go to the reviews' is a """,  # major accuracy/mistranslation error""",
    """'etc.,' is a """,  # minor style/awkward error""",
    """'involvement' is a """,  # major accuracy/mistranslation error""",
    """'the account holder' is a """,  # major accuracy/omission error""",
    """'wäre' is a """,  # minor fluency/grammar error""",
    """'dir' is a """,  # minor fluency/register error""",
    """'ve Vídni' is a """,  # major accuracy/addition error""",
    """'the stop-start' is a """,  # major accuracy/omission error""",
    """'partaje' is a """,  # minor terminology/inappropriate for context error""",
    """'' is a""",
]

icl_refinements = [
    {
        "role": "user",
        "content": """You are correcting translation given source, translation and feedback. Only return improved translation without giving additional feedback. Source: ```大众点评乌鲁木齐家居商场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评``` Translation: ```Urumqi Home Furnishing Store Channel provides you with the latest bussiness information such as the address, telephone number, bussiness hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.``` Feedback: 'of high-speed rail' is a critical accuracy/addition error\n'go to the reviews' is a major accuracy/mistranslation error\n'etc.,' is a minor style/awkward error\n Corrected translation:""",
    },
    {
        "role": "assistant",
        "content": """Dianping Urumqi Renovation and Design Channel will provide you with the address, phone number, operation time and other information of HSR Easyhome, and please come to Dianping if you are looking for a renovation company.\n\n""",
    },
    {
        "role": "user",
        "content": "Source: ```I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.``` Translation: ```Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.``` Feedback: 'involvement' is a major accuracy/mistranslation error\n'the account holder' is a major accuracy/omission error\n'wäre' is a minor fluency/grammar error\n'dir' is a minor fluency/register error\n Corrected translation:",
    },
    {
        "role": "assistant",
        "content": """Ich bitte um Entschuldigung, aber wir benötigen das Einverständnis des Kontoinhabers, um eine Bestellung mit einer anderen Person zu besprechen, falls es schon eingeholt wurde, entschuldige ich mich, aber ich kann dies ohne das Einverständnis des Kontoinhabers nicht mit Ihnen besprechen.\n\n""",
    },
    {
        "role": "user",
        "content": "Source: ```Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.``` Translation: ```Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemže obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.``` Feedback: 've Vídni' is a major accuracy/addition error\n'the stop-start' is a major accuracy/omission error\n'partaje' is a minor terminology/inappropriate for context error\n Corrected translation:",
    },
    {
        "role": "assistant",
        "content": """Ve Vídni byly obnoveny rozhovory o oživení jaderného paktu a obě strany se snaží odhadnout, jaké jsou vyhlídky na úspěch po posledních výměnách názorů v rámci přerušených jednání.\n\n""",
    },
]


@click.command()
@click.option("-lang_dir", help="zh-en, en-de")
@click.option("-task_type", help="trans, feedback or refinement")
@click.option("-batch_size", type=int)
@click.option("-suffix")
@click.option("-src_file")
@click.option("-out_file")
@click.option("-feedback_file")
def main(lang_dir, task_type, batch_size, suffix, src_file, out_file, feedback_file):
    msg_ls = []
    cor_index_gen_ls = []
    err_index_gen_ls = []
    err_index = []
    if task_type == "trans":
        if lang_dir == "zh-en":
            source_lang = "Chinese"
            tgt_lang = "English"
        elif lang_dir == "en-de":
            source_lang = "English"
            tgt_lang = "German"
        elif lang_dir == "yor-en":
            source_lang = "Yoruba"
            tgt_lang = "English"
        else:
            print("Language direction is not supported!")
            exit(1)

        src_lines = open(src_file, "r").readlines()
        src_lines = [line[:-1] for line in src_lines]
        for line in src_lines:
            message = icl_trans[lang_dir] + [
                {
                    "role": "user",
                    "content": f"{source_lang}: '{line}' {tgt_lang}: ",
                }
            ]
            msg_ls += [message]
        err_index = list(range(len(msg_ls)))
    elif task_type == "feedback":
        src_lines = open(src_file, "r").readlines()
        src_lines = [line[:-1] for line in src_lines]
        out_lines = open(out_file, "r").readlines()
        out_lines = [line[:-1] for line in out_lines]
        for src, out in zip(src_lines, out_lines):
            message = icl_eval + [
                {
                    "role": "user",
                    "content": f"Source: ```{src}``` Translation: ```{out}``` MQM annotations:",
                }
            ]
            msg_ls += [message]
        err_index = list(range(len(msg_ls)))
    elif task_type == "refinement":
        src_lines = open(src_file, "r").readlines()
        src_lines = [line[:-1] for line in src_lines]
        out_lines = open(out_file, "r").readlines()
        out_lines = [line[:-1] for line in out_lines]
        feedback_lines = open(feedback_file, "r").readlines()
        feedback_lines = [line[:-1] for line in feedback_lines]
        for index, (src, out, feedback) in enumerate(
            zip(src_lines, out_lines, feedback_lines)
        ):
            if int(feedback.split("\t")[-1]) < 0:
                feedback_str = "\n".join(feedback.split("\t")[:-1]) + "\n"
                message = icl_refinements + [
                    {
                        "role": "user",
                        "content": f"Source: ```{src}``` Translation: ```{out}``` Feedback: {feedback_str} Corrected translation:",
                    }
                ]
                msg_ls += [message]
                err_index += [index]
            else:
                cor_index_gen_ls += [[index, out]]
    else:
        print(f"{task_type} is not supported")
        exit(1)

    with open(
        f"model_outputs/mistreal/mistreal_{task_type}_{lang_dir}_{suffix}.txt", "w"
    ) as savefile:
        with tqdm(total=int(len(msg_ls) / batch_size) + 1) as pbar:
            format_str_ls = []
            for message in msg_ls:
                formatted_str = tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                format_str_ls += [formatted_str]

            for format_batch, index_batch in zip(
                batchify(format_str_ls, batch_size), batchify(err_index, batch_size)
            ):
                encodeds = tokenizer(
                    format_batch,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True,
                    padding=True,
                )["input_ids"]
                model_inputs = encodeds.to(device_id)
                with torch.no_grad():
                    generated_ids = model.generate(
                        model_inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                decoded = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                if task_type == "trans":
                    dec_ls = [
                        ele.split("[/INST]")[-1].split("\n")[0] + "\n"
                        for ele in decoded
                    ]
                elif task_type == "feedback":
                    dec_ls = []
                    for ele in decoded:
                        process_txt = ele.split("[/INST]")[-1]
                        for icl_instance in ignore_outputs:
                            process_txt = process_txt.replace(
                                icl_instance, "[PLACEHOLDER]"
                            )
                        if "error" in process_txt:
                            score = (
                                process_txt.count("' is a critical ") * (-10)
                                + process_txt.count("' is a major ") * (-5)
                                + process_txt.count("' is a minor ") * (-1)
                            )
                        else:
                            score = 0
                        anno_str = ""
                        if score < 0:
                            for anno in process_txt.split("\n"):
                                if (
                                    "' is a critical " in anno
                                    or "' is a major " in anno
                                    or "' is a minor " in anno
                                ) and "error" in anno:
                                    anno_str += anno + "\t"
                        else:
                            anno_str = "no-error"
                        dec_ls += [anno_str + "\t" + str(score) + "\n"]
                elif task_type == "refinement":
                    for ele in decoded:
                        print(ele.split("Corrected translation:")[-1].strip())
                        print("-" * 50)
                else:
                    print("Task type is not supported!")
                    exit(1)

                for dec in dec_ls:
                    savefile.write(dec)
                pbar.update(1)
    print(
        f"model_outputs/mistreal/mistreal_{task_type}_{lang_dir}_{suffix}.txt is saved!"
    )


if __name__ == "__main__":
    main()
