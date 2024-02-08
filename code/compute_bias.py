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
    denominator = np.sum(np.abs(np.add.outer(X, X) - 2 * theta))

    # handle the case when Pr(X=theta) = 1
    if denominator == 0:
        return 0
    else:
        return 1 - numerator / denominator

def skewness(X):
    return sum([ele**3 for ele in X])/len(X) / (sum([ele**2 for ele in X])/len(X))**(3/2)


@click.command()
@click.option("-bleurt_nor_file")
@click.option("-llm_score_file")
def main(bleurt_nor_file, llm_score_file):
    lines = open(
        llm_score_file,
        "r",
    ).readlines()
    final_ls = "".join(lines).split("[SEP_TOKEN_WENDA]")[:-1]
    score_ls = []
    for ele in final_ls:
        score_ls += [
            max(
                -25,
                (
                    -1 * ele.count("minor") + -5 * ele.count("major") + (-5) * ele.count("critical")
                ),
            )
        ]
    mapped_model = open(
        bleurt_nor_file, "r"
    ).readlines()
    mapped_model = [float(ele) for ele in mapped_model]
    print("Model Score: ", sum(score_ls)/len(score_ls))
    print("BLEURT Score: ", sum(mapped_model)/len(mapped_model))
    # 1) Out of bia defination
    diff_ls = [m_score - g_score for m_score, g_score in zip(score_ls, mapped_model)]
    mean_bias = sum(diff_ls) / len(diff_ls)
    print("Bias mean: ", mean_bias)

    # 2) Use skewness to find the bias out of statistical distribution
    skewsize = skewness(diff_ls)
    print("Bias skewness: ", skewsize)

    # 3) Study the shape of distribution
    d_skew = distance_skewness(diff_ls, 0)
    print("Bias distance skewness: ", d_skew)
    print()


if __name__ == "__main__":
    main()
