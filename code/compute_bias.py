import scipy.stats as stats
import numpy as np
from scipy import interpolate
from matplotlib import pyplot
import click

# we have three formula to compute bias


# This implementation is based on Wikipedia description
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


@click.command()
@click.option("-model_name", help="Assuming all model outputs are in the MQM styles")
@click.option("-eval_name")
@click.option("-lang")
def main(model_name, eval_name, lang):
    file_name = f"model_outputs/{model_name}/yor-en_eval_100_one-shot_{eval_name}.txt"
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
    mapped_model = open(
        f"model_outputs/{eval_name}/{lang}_base_outputs_{eval_name}_bleurt_nor.txt", "r"
    ).readlines()
    mapped_model = [float(ele) for ele in mapped_model]

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


if __name__ == "__main__":
    main()
