import numpy as np
from scipy import interpolate
import click


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


@click.command()
@click.option(
    "-human_file",
    help="/mnt/data3/wendaxu/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/human-scores/zh-en.mqm.seg.score",
)
@click.option(
    "-obs_score_file",
    help="/mnt/data3/wendaxu/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/metric-scores/zh-en/BLEURT-20-refA.seg.score",
)
@click.option(
    "-new_score_file",
    help="",
)
@click.option("-save_name")
def main(human_file, obs_score_file, new_score_file, save_name):
    # ref_lines = open(human_file, "r").readlines()
    # ref_dict = from_list_to_dict(ref_lines)

    # out_lines = open(obs_score_file, "r").readlines()
    # out_dict = from_list_to_dict(out_lines)

    # ref_score_ls = []
    # out_score_ls = []

    # for ref_sys in set(ref_dict) - {"refA"}:
    #     ref_score_ls += ref_dict[ref_sys]
    #     out_score_ls += out_dict[ref_sys]
    ref_score_ls = open(human_file, "r").readlines()
    ref_score_ls = [float(ele[:-1]) for ele in ref_score_ls]

    out_score_ls = open(obs_score_file, "r").readlines()
    out_score_ls = [float(ele[:-1]) for ele in out_score_ls]

    obs_data = np.array(ref_score_ls)
    model_data = np.array(out_score_ls)

    # Sorted observation data
    sorted_observed = np.sort(obs_data)

    # Sorted model data
    sorted_model = np.sort(model_data)

    print(len(sorted_observed))
    print(len(sorted_model))
    # Interpolation function
    f = interpolate.interp1d(
        sorted_model, sorted_observed, bounds_error=False, fill_value="extrapolate"
    )

    # Mapping the model data to observation
    gt_other = open(new_score_file, "r").readlines()
    gt_other = [float(score[:-1]) for score in gt_other]
    mapped_model = [max(-25, score) for score in f(gt_other)]
    print("Average score: ", sum(mapped_model) / len(mapped_model))
    mapped_model = [str(ele) + "\n" for ele in mapped_model]
    with open(save_name, "w") as f:
        f.writelines(mapped_model)


if __name__ == "__main__":
    main()
