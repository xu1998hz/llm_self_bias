import numpy as np
from scipy import interpolate

ref_lines = open(
    "/mnt/data3/wendaxu/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/human-scores/zh-en.mqm.seg.score",
    "r",
).readlines()


def from_list_to_dict(ref_lines):
    ref_dict = {}
    for ref in ref_lines:
        sys_name = ref.split("\t")[0]
        score = ref[:-1].split("\t")[1]
        if sys_name not in ref_dict and score != "None":
            ref_dict[sys_name] = []
        if score != "None":
            ref_dict[sys_name] += [float(score)]
    return ref_dict


ref_dict = from_list_to_dict(ref_lines)

out_lines = open(
    "/mnt/data3/wendaxu/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/metric-scores/zh-en/BLEURT-20-refA.seg.score",
    "r",
).readlines()

out_dict = from_list_to_dict(out_lines)

ref_score_ls = []
out_score_ls = []

for ref_sys in set(ref_dict) - {"refA"}:
    ref_score_ls += ref_dict[ref_sys]
    out_score_ls += out_dict[ref_sys]

obs_data = np.array(ref_score_ls)
model_data = np.array(out_score_ls)

# Sorted observation data
sorted_observed = np.sort(obs_data)

# Sorted model data
sorted_model = np.sort(model_data)

# Interpolation function
f = interpolate.interp1d(
    sorted_model, sorted_observed, bounds_error=False, fill_value="extrapolate"
)

# Mapping the model data to observation

for i in range(1, 11):
    for other_sys in [
        # "gpt-3.5-turbo",
        "gpt-4",
        # "gemini",
    ]:
        gt_other = open(
            # f"/mnt/data3/wendaxu/self-improve/model_outputs/{other_sys}/yor-en_base_outputs_{other_sys}_bleurt.txt",
            f"model_outputs/{other_sys}/yor-en_self_refinement_100_{other_sys}_new_{i}_rerun_bleurt.txt",
            "r",
        ).readlines()
        gt_other = [float(score[:-1]) for score in gt_other]
        mapped_model = [max(-25, score) for score in f(gt_other)]
        print(i)
        print(other_sys)
        print(sum(mapped_model) / len(mapped_model))
        print()

# file_name = f"/mnt/data3/wendaxu/self-improve/model_outputs/{other_sys}/yor-en_eval_100_one-shot_refined_2nd_gpt-4.txt"
# lines = open(
#     file_name,
#     "r",
# ).readlines()
# final_ls = "".join(lines).split("[SEP_TOKEN_WENDA]")[:-1]
# print(file_name)
# print(len(final_ls))
# score_ls = []
# for ele in final_ls:
#     score_ls += [
#         max(
#             -25,
#             (
#                 -1 * ele.count("minor")
#                 + -5 * ele.count("major")
#                 + (-5) * ele.count("critical")
#             ),
#         )
#     ]

# import matplotlib.pyplot as plt
# import numpy as np

# x = np.array(mapped_model)
# y = np.array(score_ls)

# plt.scatter(x, y)
# plt.savefig("gpt-4_2.png")
