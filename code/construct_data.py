from datasets import load_dataset
import click

@click.command()
@click.option('-lang')
def main(lang):
    data = load_dataset('Muennighoff/flores200', lang)
    src_ls = [ele['sentence']+'\n' for ele in data['devtest']]
    src_ls = src_ls[:100]

    with open(f'srcs/{lang}-en_src_100.txt', 'w') as f:
        f.writelines(src_ls)
        print(f'srcs/{lang}-en_src_100.txt is saved!')

if __name__ == "__main__":
    main()