import glob
from datasets import load_dataset
import json
import math


# for file_name in glob.glob("model_outputs/gpt-4/sci_eval_100*"):
#     final_str = " ".join(open(file_name, "r").readlines())
#     count = final_str.count("True")
#     total = final_str.count("True") + final_str.count("False")
#     print(file_name)
#     print(total)
#     print(count / total)
#     print()


# def is_float(element: any) -> bool:
#     # If you expect None to be passed:
#     if element is None:
#         return False
#     try:
#         float(element)
#         return True
#     except ValueError:
#         return False


# model_name = "gpt-4"
# lines = open(
#     f"model_outputs/{model_name}/sci_self_refinement_100_{model_name}_refined_1.txt",
#     "r",
# ).readlines()
# final_ls = []
# for line in lines:
#     if "#" in line:
#         ele = line.split("#")[-1].strip()
#         if is_float(ele):
#             final_ls += [float(ele)]
#         else:
#             final_ls += [None]
#     else:
#         if len(line.split()) > 1:
#             ele = line.split()[-1][:-1]
#             if is_float(ele):
#                 final_ls += [float(ele)]
#             else:
#                 final_ls += [None]

# gt_ls = []
# for file_name in glob.glob("scibench/dataset/original/*_sol.json"):
#     data = json.load(open(file_name))
#     gt_ls += [float(ele["answer_number"]) for ele in data[1:]]

# results = []
# for ans, gt in zip(final_ls, gt_ls):
#     if ans:
#         if ans >= 1:
#             ind = math.isclose(ans, gt, abs_tol=0.1)
#         else:
#             ind = math.isclose(ans, gt, rel_tol=0.1)
#         if ind:
#             results += [1]
#         else:
#             results += [0]
#     else:
#         results += [0]

# print(len(final_ls))
# print(len(gt_ls))
# print(sum(results) / len(results))


# lines = open("model_outputs/gpt-4/math_base_outputs_gpt-4.txt", "r").readlines()
# data = load_dataset("gsm8k", "main")["test"]["answer"][:100]
# count = 0
# for answer, line in zip(data, lines):
#     if line[:-1].split("#")[-1].strip() == answer.split("#")[-1].strip():
#         count += 1
# print(count / len(data))

for file_name in glob.glob(
    "/mnt/data3/wendaxu/self-improve/model_outputs/gemini/yor-en_eval*gpt-4*"
):
    lines = open(
        file_name,
        "r",
    ).readlines()
    final_ls = "".join(lines).split("[SEP_TOKEN_WENDA]")[:-1]
    print(file_name)
    print(len(final_ls))
    score = 0
    for ele in final_ls:
        score += (
            -1 * ele.count("minor")
            + -5 * ele.count("major")
            + (-5) * ele.count("critical")
        )

    print(score / 100)

# for file_name in glob.glob(
#     "/mnt/data3/wendaxu/self-improve/model_outputs/mistreal/*feedback_yor-en*"
# ):
#     lines = open(
#         file_name,
#         "r",
#     ).readlines()
#     print(file_name)
#     print(len(lines))
#     score = 0
#     for ele in lines:
#         score += (
#             -1 * ele.count("minor")
#             + -5 * ele.count("major")
#             + (-10) * ele.count("critical")
#         )

#     print(score / 100)

# from comet import download_model, load_from_checkpoint

# model_path = download_model("Unbabel/wmt22-comet-da")
# model = load_from_checkpoint(model_path)

# import evaluate

# # sacrebleu = evaluate.load("sacrebleu")
# sacrebleu = evaluate.load("chrf")


#         print(cur_addr)
#         print(sum(scores_ls) / len(scores_ls))
#         print()

#         # results = sacrebleu.compute(predictions=out_lines, references=ref_lines)
#         # print(results)

#         # data = [
#         #     {"src": src, "mt": out, "ref": ref}
#         #     for src, out, ref in zip(src_lines, out_lines, ref_lines)
#         # ]
#         # model_output = model.predict(data, batch_size=8, gpus=1)

#         # print(model_output.system_score)
