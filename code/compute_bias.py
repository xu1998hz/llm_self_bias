import scipy.stats as stats
import numpy as np
from scipy import interpolate
from matplotlib import pyplot

# we have three formula to compute bias
# we first find a quantile mapping function which maps metric score into human score range

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


def distance_skewness(X, theta):
    # pairwise distances between the elements of X
    pairwise_distances = np.abs(np.subtract.outer(X, X))

    # numerator and denominator of the distance skewness formula
    numerator = np.sum(pairwise_distances)
    denominator = np.sum(np.abs(np.subtract.outer(X, X) - 2 * theta))

    # handle the case when Pr(X=theta) = 1
    if numerator == 0 and denominator == 0:
        return 0
    else:
        return 1 - numerator / denominator


# Mapping the model data to observation
other_sys = "gpt-4"
diff_ls_all = []
for i in range(1, 11):
    gt_other = open(
        # f"/mnt/data3/wendaxu/self-improve/model_outputs/{other_sys}/yor-en_base_outputs_{other_sys}_bleurt.txt",
        f"model_outputs/{other_sys}/yor-en_base_outputs_gpt-4_bleurt.txt",
        "r",
    ).readlines()
    gt_other = [float(score[:-1]) for score in gt_other]
    mapped_model = [max(-25, score) for score in f(gt_other)]
    print(sum(mapped_model) / len(mapped_model))

    file_name = f"/mnt/data3/wendaxu/self-improve/model_outputs/{other_sys}/yor-en_eval_100_one-shot_gpt-4.txt"
    lines = open(
        file_name,
        "r",
    ).readlines()
    final_ls = "".join(lines).split("[SEP_TOKEN_WENDA]")[:-1]
    score_ls = []
    for ele in final_ls:
        score_ls += [
            max(
                -25,
                (
                    -1 * ele.count("minor")
                    + -5 * ele.count("major")
                    + (-10) * ele.count("critical")
                ),
            )
        ]
    print(sum(score_ls) / len(score_ls))

    # 1) Out of bia defination
    diff_ls = [m_score - g_score for m_score, g_score in zip(score_ls, mapped_model)]
    mean_bias = sum(diff_ls) / len(diff_ls)
    print("Bias mean: ", mean_bias)

    # 2) Use skewness to find the bias out of statistical distribution
    skewsize = stats.skew(diff_ls, bias=True)
    print("Bias skewness: ", skewsize)

    # 3) Study the shape of distribution
    d_skew = distance_skewness(diff_ls, mean_bias)
    print("Bias distance skewness: ", d_skew)
    print()

    diff_ls_all += [diff_ls]

bins = np.linspace(-25, 25, 100)

pyplot.hist(
    [ele - 0.5 for ele in diff_ls_all[0]], bins, alpha=0.5, label="1st iteration"
)
pyplot.hist(diff_ls_all[-1], bins, alpha=0.5, label="10th iteration")
pyplot.legend(loc="upper left")
pyplot.savefig("example.png")
