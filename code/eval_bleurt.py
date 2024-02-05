from bleurt import score
import click


@click.command()
@click.option("-lang_dir", help="yor-en, en-de or zh-en")
@click.option("-file_name")
@click.option("-save_name")
def main(file_name, lang_dir, save_name):
    scorer = score.BleurtScorer("BLEURT-20")
    ref_lines = open(
        f"refs/{lang_dir}_ref_100.txt", "r"
    ).readlines()
    ref_lines = [ele[:-1] for ele in ref_lines]

    out_lines = open(file_name, "r").readlines()
    out_lines = [ele[:-1] for ele in out_lines]
    scores_ls = scorer.score(references=ref_lines, candidates=out_lines, batch_size=16)
    scores_ls = [str(score) + "\n" for score in scores_ls]
    save_file = open(save_name, "w")
    save_file.writelines(scores_ls)


if __name__ == "__main__":
    main()
