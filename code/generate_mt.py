from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")
model = model.to("cuda")
save_file = open("nllb.txt", "w")
src_lines = open(
    "/mnt/data3/wendaxu/self-improve/srcs/yor-en_src_100.txt", "r"
).readlines()
src_lines = [ele[:-1] for ele in src_lines]
for ele in src_lines:
    inputs = tokenizer(ele, return_tensors="pt").to("cuda")

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"],
        max_length=512
    )
    save_file.write(
        tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0] + "\n"
    )
