import statistics as sts
import matplotlib.pyplot as plt
import json

from rex.utils.io import dump_jsonlines, load_jsonlines


def load_tree_data(
    tree_filepath,
    instruction_quality: float = 0.6,
    response_quality: float = 0.6,
    instruction_word_num: int = 5,
    response_word_num: int = 5,
    lang: str = "en",
    response_rank: int = 0,
    first_turn: bool = False,
):
    trees = load_jsonlines(tree_filepath)
    pairs = []
    resp_scores = []

    def _traverse(ins: dict):
        for reply in ins["replies"]:
            if (
                ins.get("lang") == lang
                and reply.get("lang") == lang
                and reply.get("rank") == response_rank
            ):
                inst_qlt = ins["labels"].get("quality", {"value": 1.0})["value"]
                resp_qlt = reply["labels"].get("quality", {"value": 0.0})["value"]
                resp_scores.append(resp_qlt)
                if inst_qlt > instruction_quality and resp_qlt > response_quality:
                    if (
                        len(ins["text"].split()) > instruction_word_num
                        and len(reply["text"].split()) > response_word_num
                    ):
                        pairs.append(
                            {
                                "instruction": ins["text"],
                                "instruction_quality": inst_qlt,
                                "response": reply["text"],
                                "response_quality": resp_qlt,
                            }
                        )
        # TODO: why this?
        if not first_turn:
            for reply in ins["replies"]:
                _traverse(reply)

    for tree in trees:
        prompt = tree["prompt"]
        _traverse(prompt)

    return pairs, resp_scores


if __name__ == "__main__":
    first_turn = True
    instruction_quality = -1
    response_quality = -1
    dump_num = 3200
    pairs, resp_scpres = load_tree_data(
        "2023-04-12_oasst_ready.trees.jsonl", 
        instruction_quality=instruction_quality, 
        response_quality=response_quality, 
        first_turn=first_turn
    )

    print(f"#data: {len(pairs)}, #dump: {dump_num}")
    pairs.sort(
        key=lambda ins: ins["instruction_quality"] + ins["response_quality"],
        reverse=True,
    )
    data = pairs[:dump_num]
    # instruction_lens = []
    # response_lens = []
    # for ins in dump_data:
    #     instruction_lens.append(len(ins["instruction"]))
    #     response_lens.append(len(ins["response"]))
    # print(
    #     f"Instruction len: {sts.mean(instruction_lens):.0f}±{sts.stdev(instruction_lens):.0f}, "
    #     f"Response len: {sts.mean(response_lens):.0f}±{sts.stdev(response_lens):.0f}"
    # )
    # dump_jsonlines(dump_data, "seed_first_turn.jsonl")

    # f = "seed_first_turn.jsonl"
    # with open(f, 'r') as file:
    #     data = file.readlines()
    #     data = [json.loads(line) for line in data]

    data_dict = []

    for i in range(len(data)):
        d = {}
        d['input'] = data[i]['instruction']
        d['output'] = data[i]['response']
        # d['instruction_quality'] = data[i]['instruction_quality']
        # d['response_quality'] = data[i]['response_quality']
        data_dict.append(d)

    json_data = {"type": "text2text"}
    json_data['instances'] = data_dict

    with open('ift_seed.json', 'w') as file:
        json.dump(json_data, file, indent=4)
