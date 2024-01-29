from bleurt import score

scorer = score.BleurtScorer("BLEURT-20")

src_lines = open(
    "/mnt/data3/wendaxu/self-improve/srcs/yor-en_src_100.txt", "r"
).readlines()
src_lines = [ele[:-1] for ele in src_lines]
for lang_dir in ["yor-en"]:  # "en-de", "zh-en",
    ref_lines = open(
        f"/mnt/data3/wendaxu/self-improve/refs/{lang_dir}_ref_100.txt", "r"
    ).readlines()
    ref_lines = [ele[:-1] for ele in ref_lines]
    for model_type in ["gpt-4"]:  # "gemini",
        for cur_addr in [
            f"model_outputs/{model_type}/yor-en_self_refinement_100_{model_type}_new_8_rerun.txt",
            f"model_outputs/{model_type}/yor-en_self_refinement_100_{model_type}_new_9_rerun.txt",
            f"model_outputs/{model_type}/yor-en_self_refinement_100_{model_type}_new_10_rerun.txt",
            # f"model_outputs/{model_type}/yor-en_self_refinement_100_{model_type}_new_4_rerun.txt",
            # f"model_outputs/{model_type}/yor-en_self_refinement_100_{model_type}_new_5_rerun.txt",
            # f"model_outputs/{model_type}/yor-en_self_refinement_100_{model_type}_new_6_rerun.txt",
            # f"model_outputs/{model_type}/yor-en_self_refinement_100_{model_type}_new_7_rerun.txt",
        ]:
            print(cur_addr)
            out_lines = open(cur_addr, "r").readlines()
            out_lines = [ele[:-1] for ele in out_lines]
            scores_ls = scorer.score(
                references=ref_lines, candidates=out_lines, batch_size=16
            )
            scores_ls = [str(score) + "\n" for score in scores_ls]
            save_file = open(cur_addr.replace(".txt", "_bleurt.txt"), "w")
            save_file.writelines(scores_ls)
