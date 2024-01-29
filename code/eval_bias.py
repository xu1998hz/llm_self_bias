import numpy as np
import pandas as pd
import scipy.stats as stats

designated_sys = "gpt-3.5-turbo"
measure_sys = "gemini"
sys_dict = {
    "gpt-3.5-turbo": {
        "gpt-3.5-turbo": ["gpt-4", "gemini"],
        "gpt-4": ["gemini"],
        "gemini": ["gpt-4"],
    },
    "gpt-4": {
        "gpt-4": ["gpt-3.5-turbo", "gemini"],
        "gpt-3.5-turbo": ["gemini"],
        "gemini": ["gpt-4"],
    },
    "gemini": {
        "gemini": ["gpt-3.5-turbo", "gpt-4"],
        "gpt-3.5-turbo": ["gpt-4"],
        "gpt-4": ["gpt-3.5-turbo"],
    },
}
cur_dict = {
    "greater": {"preds": [], "subgroup": []},
    "less": {"preds": [], "subgroup": []},
    "tie": {"preds": [], "subgroup": []},
}
for sys_index, sys in enumerate(sys_dict[measure_sys]):
    file_name_main = f"/mnt/data3/wendaxu/self-improve/model_outputs/{designated_sys}/yor-en_eval_100_one-shot_{sys}.txt"
    gt_main = open(
        f"/mnt/data3/wendaxu/self-improve/model_outputs/{sys}/yor-en_base_outputs_{sys}_bleurt.txt"
    ).readlines()
    gt_main = [round(float(score[:-1]), 2) for score in gt_main]
    other_sys_ls = sys_dict[measure_sys][sys]

    for other_sys in other_sys_ls:
        main_lines = "".join(
            open(
                file_name_main,
                "r",
            ).readlines()
        ).split(
            "[SEP_TOKEN_WENDA]"
        )[:-1]

        sec_name = f"/mnt/data3/wendaxu/self-improve/model_outputs/{designated_sys}/yor-en_eval_100_one-shot_{other_sys}.txt"
        gt_other = open(
            f"/mnt/data3/wendaxu/self-improve/model_outputs/{other_sys}/yor-en_base_outputs_{other_sys}_bleurt.txt",
            "r",
        ).readlines()
        gt_other = [round(float(score[:-1]), 2) for score in gt_other]
        other_lines = " ".join(
            open(
                sec_name,
                "r",
            ).readlines()
        ).split(
            "[SEP_TOKEN_WENDA]"
        )[:-1]

        for main_ele, other_ele, gt_main_ele, gt_other_ele in zip(
            main_lines, other_lines, gt_main, gt_other
        ):
            cur_main_score = (
                -1 * main_ele.count("minor")
                + -5 * main_ele.count("major")
                + (-10) * main_ele.count("critical")
            )
            cur_other_score = (
                -1 * other_ele.count("minor")
                + -5 * other_ele.count("major")
                + (-10) * other_ele.count("critical")
            )
            if gt_main_ele > gt_other_ele:
                prefix = "greater"
            elif gt_main_ele < gt_other_ele:
                prefix = "less"
            else:
                prefix = "tie"

            if cur_main_score > cur_other_score:
                cur_dict[prefix]["preds"] += [0]
            elif cur_main_score < cur_other_score:
                cur_dict[prefix]["preds"] += [1]
            else:
                cur_dict[prefix]["preds"] += [2]

            cur_dict[prefix]["subgroup"] += [sys_index]

v_list = []
for cur_dict in cur_dict.values():
    if len(cur_dict["preds"]) > 0:
        df = pd.DataFrame(
            {"predictions": cur_dict["preds"], "subgroups": cur_dict["subgroup"]}
        )
        crosstab = pd.crosstab(df.subgroups, df.predictions)
        chi2 = stats.chi2_contingency(crosstab)[0]
        print(crosstab)
        print(stats.chi2_contingency(crosstab))
        dof = min(crosstab.shape) - 1
        n = crosstab.sum().sum()
        v = np.sqrt(chi2 / (n * dof))
        v_list.append(v)

v_values = np.asarray(v_list)
# When a model predicts correctly all examples
# in a given class across all subgroups
# dof=0 and the corresponding v is NaN.
# We remove NaNs before computing skewsize.
v_values = v_values[~np.isnan(v_values)]
print(v_values)
skewsize = stats.skew(v_values)
print(skewsize)
