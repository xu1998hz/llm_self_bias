from mt_metrics_eval import data
from scipy import stats
from itertools import combinations

level = "seg"

metric_ls = [
    f"XCOMET-XXL-refA.{level}.score",
    f"CometKiwi-src.{level}.score",
    f"sescoreX-refA.{level}.score",
    f"instructscore-refA.{level}.score",
    f"mbr-metricx-qe-src.{level}.score",
    f"BLEURT-20-refA.{level}.score",
    f"MaTESe-refA.{level}.score",
    f"MetricX-23-refA.{level}.score",
    f"GEMBA-MQM-src.{level}.score",
    f"BLEU-refA.{level}.score",
    f"BERTscore-refA.{level}.score",
    f"chrF-refA.{level}.score",
    f"tokengram_F-refA.{level}.score",
    f"YiSi-1-refA.{level}.score",
]

sys_pair_dict = {"zh-en": "Lan-BridgeMT", "en-de": "ONLINE-W", "he-en": "ONLINE-A"}

compare_type = "gpt4_all"  # "all"  # 'gpt4_top_system' "gpt4_all"  "all"

for lang in ["he-en"]:  # , "he-en", "en-de"
    evs = data.EvalSet("wmt23", lang)
    mqm_scores = evs.Scores(level, "mqm")
    sys_pairs_ls = []
    print(lang)

    for metric_name in metric_ls:
        cur_addr = f"/mnt/data3/wendaxu/.mt-metrics-eval/mt-metrics-eval-v2/wmt23/metric-scores/{lang}/{metric_name}"
        lines = open(cur_addr, "r").readlines()
        metric_dict = {}
        for line in lines:
            name, score = line[:-1].split("\t")[0], line[:-1].split("\t")[1]
            if name not in metric_dict:
                metric_dict[name] = []
            metric_dict[name] += [float(score)]

        print(len(set(evs.sys_outputs) - {"GPT4-5shot"}))

        if compare_type == "gpt4_top_system":
            sys_pairs = [["GPT4-5shot", sys_pair_dict[lang]]]
        elif compare_type == "gpt4_all":
            sys_pairs = [["ONLINE-A", sys_name] for sys_name in set(evs.sys_outputs)]
        elif compare_type == "all":
            sys_pairs = []
            for sys_pair in list(
                combinations(set(evs.sys_outputs) - {"GPT4-5shot"}, 2)
            ):
                sys_pairs += [sys_pair]
        else:
            print(f"Your type: {compare_type} is not supported!")
            exit(1)

        better_p, better_n, worse_p, worse_n, tie_p, tie_n = 0, 0, 0, 0, 0, 0
        bp, bn, wp, wn, tp, tn = 0, 0, 0, 0, 0, 0
        for sys_pair in sys_pairs:
            if sys_pair[0] != "refA" and sys_pair[1] != "refA":
                for h_sys1_score, h_sys2_score, g_sys1_score, g_sys2_score in zip(
                    mqm_scores[sys_pair[0]],
                    mqm_scores[sys_pair[1]],
                    metric_dict[sys_pair[0]],
                    metric_dict[sys_pair[1]],
                ):
                    if h_sys1_score != None:
                        if (
                            h_sys1_score - h_sys2_score > 0
                            and g_sys1_score > g_sys2_score
                        ):
                            better_p += 1
                        elif (
                            h_sys1_score - h_sys2_score < 0
                            and g_sys1_score < g_sys2_score
                        ):
                            worse_p += 1
                        elif (
                            h_sys1_score == h_sys2_score
                            and g_sys1_score == g_sys2_score
                        ):
                            tie_p += 1
                        elif (
                            h_sys1_score - h_sys2_score > 0
                            and g_sys1_score <= g_sys2_score
                        ):
                            better_n += 1
                        elif (
                            h_sys1_score - h_sys2_score < 0
                            and g_sys1_score >= g_sys2_score
                        ):
                            worse_n += 1
                        elif (
                            h_sys1_score == h_sys2_score
                            and g_sys1_score != g_sys2_score
                        ):
                            tie_n += 1
                        else:
                            print("Error")

        print(metric_name)
        print(
            "better_p, better_n, worse_p, worse_n, tie_p, tie_n: ",
            better_p,
            better_n,
            worse_p,
            worse_n,
            tie_p,
            tie_n,
        )
        print("Better Accuracy: ", better_p / (better_p + better_n))
        print("Worse Accuracy: ", worse_p / (worse_p + worse_n))
        print(
            "Accuracy: ",
            (better_p + worse_p) / (better_p + worse_p + better_n + worse_n),
        )
        print("Tie Accuracy: ", tie_p / (tie_p + tie_n))
        print("-" * 20)
