from scipy import stats


def ret_score_ls(file_name):
    lines = open(file_name, "r").readlines()

    score_ls = []
    count = 0
    for ele in lines:
        score = 0
        score += ele.count("critical") * (-25)
        score += ele.count("major") * (-5)
        score += ele.count("minor") * (-1)
        score_ls += [score]
        if score != 0:
            count += 1
    print(count)
    return score_ls


score_ls_1 = ret_score_ls(
    "/mnt/data3/wendaxu/self-improve/zh-en_self_eval_500_one-shot_src.txt"
)
score_ls_2 = ret_score_ls("zh-en_self_eval_500_one-shot_pseudo_ref_2.txt")
print(sum(score_ls_1) / len(score_ls_1))
print(sum(score_ls_2) / len(score_ls_2))
res = stats.kendalltau(score_ls_1, score_ls_2)
print("Kendall: ", res)
res = stats.pearsonr(score_ls_1, score_ls_2)
print("Pearson: ", res)
