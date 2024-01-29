import glob
import json

out_lines = []
for file_name in glob.glob("scibench/dataset/original/*_sol.json"):
    data = json.load(open(file_name))
    for ele in data[1:]:
        label = ele["answer_number"]
        out_lines += [
            ele["solution"].replace("\n", "\t")
            + "\t"
            + f"####{label}"
            + "[SEP_TOKEN_WENDA]"
        ]

with open(f"refs/sci_ref.txt", "w") as f:
    f.writelines(out_lines)
    print(f"refs/sci_ref.txt is saved!")
print(len(out_lines))
