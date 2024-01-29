import glob
import math
import json


# def is_float(element: any) -> bool:
#     # If you expect None to be passed:
#     if element is None:
#         return False
#     try:
#         float(element)
#         return True
#     except ValueError:
#         return False


# model_name_eval = "gpt-3.5-turbo"
# print("eval:", model_name_eval)
# for model_name_out in ["gpt-3.5-turbo", "gpt-4", "gemini"]:
#     for sys, mode in zip(
#         ["human"],  #
#         ["base_outputs"],  # "self_refinement_100"
#     ):
#         lines = "".join(
#             open(
#                 "refs/sci_ref.txt",  # f"model_outputs/{model_name_out}/sci_{mode}_{sys}.txt",
#                 "r",
#             ).readlines()
#         ).split("[SEP_TOKEN_WENDA]")[:-1]
#         print(len(lines))
#         final_ls = []
#         for line in lines:
#             if "#" in line:
#                 ele = line.split("#")[-1].strip()
#                 if is_float(ele):
#                     final_ls += [float(ele)]
#                 else:
#                     final_ls += [None]
#             else:
#                 if len(line.split()) > 1:
#                     ele = line.split()[-1][:-1]
#                     if is_float(ele):
#                         final_ls += [float(ele)]
#                     else:
#                         final_ls += [None]

#         gt_ls = []
#         for file_name in glob.glob("scibench/dataset/original/*_sol.json"):
#             data = json.load(open(file_name))
#             gt_ls += [float(ele["answer_number"]) for ele in data[1:]]

#         results = []

#         for ans, gt in zip(final_ls, gt_ls):
#             if ans:
#                 if ans >= 1:
#                     ind = math.isclose(ans, gt, abs_tol=0.1)
#                 else:
#                     ind = math.isclose(ans, gt, rel_tol=0.1)
#                 if ind:
#                     results += [1]
#                 else:
#                     results += [0]
#             else:
#                 results += [0]

#         file_name = f"model_outputs/{model_name_eval}/sci_eval_100_one-shot_{sys}.txt"
#         eval_lines = open(file_name, "r").readlines()
#         eval_lines = "".join(eval_lines).split("[SEP_TOKEN_WENDA]")[:-1]
#         score_ls = []
#         for eval in eval_lines:
#             if "False" in eval:
#                 score_ls += [0]
#             else:
#                 score_ls += [1]
#         # print(score_ls)
#         # print(results)
#         count = 0
#         for pred, gt in zip(score_ls, results):
#             if pred != gt:
#                 count += 1

#         print(sys)
#         print(count)
#         print(len(score_ls))
#         print(count / len(score_ls))

model_name = "gpt-4"
gt_scores = open(
    f"model_outputs/{model_name}/yor-en_self_refinement_100_{model_name}_bleurt.txt",
    "r",
).readlines()
gt_scores = [float(score[:-1]) for score in gt_scores]


out_scores = "".join(
    open(
        f"model_outputs/gpt-4/yor-en_eval_100_one-shot_gpt-4.txt",
        "r",
    ).readlines()
).split("[SEP_TOKEN_WENDA]")[:-1]
out_scores = [
    (
        max(
            -25,
            -1 * score_str.count("minor")
            + -5 * score_str.count("major")
            + (-10) * score_str.count("critical"),
        )
        + 25
    )
    / 25
    for score_str in out_scores
]

bias = 0
for gt, out in zip(gt_scores, out_scores):
    bias += abs(gt - out)
print(bias / len(gt_scores))
