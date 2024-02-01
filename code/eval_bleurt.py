from bleurt import score
import click


@click.command()
@click.option("-lang_dir", help="yor-en, en-de or zh-en")
@click.option("-model_type")
def main(model_type, lang_dir):
    scorer = score.BleurtScorer("BLEURT-20")
    ref_lines = open(
        f"/mnt/data3/wendaxu/peril_self_improve/refs/{lang_dir}_ref_100.txt", "r"
    ).readlines()
    ref_lines = [ele[:-1] for ele in ref_lines]

    cur_addr = f"model_outputs/{model_type}/{lang_dir}_base_outputs_{model_type}.txt"
    out_lines = open(cur_addr, "r").readlines()
    out_lines = [ele[:-1] for ele in out_lines]
    scores_ls = scorer.score(references=ref_lines, candidates=out_lines, batch_size=16)
    scores_ls = [str(score) + "\n" for score in scores_ls]
    save_file = open(cur_addr.replace(".txt", "_bleurt.txt"), "w")
    save_file.writelines(scores_ls)


if __name__ == "__main__":
    main()
