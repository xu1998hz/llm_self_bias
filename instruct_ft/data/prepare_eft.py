import statistics as sts
import matplotlib.pyplot as plt
from math import floor
import json
import random

from rex.utils.io import dump_jsonlines, load_jsonlines


def load_tree_data(
    tree_filepath,
    inst_qlt_thres: float = 0.6,
    resp_qlt_thres: float = 0.6,
    lang: str = "en",
    first_turn: bool = False,
):
    trees = load_jsonlines(tree_filepath)
    pairs = []
    resp_scores = []
    converted_scores = []

    def _traverse(ins: dict):
        for reply in ins["replies"]:
            if (ins.get("lang") == lang and reply.get("lang") == lang and "quality" in reply["labels"]):
                inst_qlt = ins["labels"].get("quality", {"value": 1.0})["value"]
                # resp_qlt = reply["labels"].get("quality", {"value": 0.0})["value"]
                resp_qlt = reply["labels"]['quality']["value"]
                resp_scores.append(resp_qlt)

                if inst_qlt > inst_qlt_thres and resp_qlt > resp_qlt_thres and len(ins["text"].split()) > 5 and len(reply["text"].split()) > 5:
                    # convert scale from 0 - 1 to 1 - 5
                    score = floor(resp_qlt * 5)
                    converted_scores.append(score)
                    inst = ins["text"]
                    resp = reply["text"]
                    # make prompt template
                    prompt = f"Review the user’s question and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:\n- Add 1 point if the response is relevant and provides some information related to the user’s inquiry, even if it is incomplete or contains some irrelevant content.\n- Add another point if the response addresses a substantial portion of the user’s question, but does not completely resolve the query or provide a direct answer.\n- Award a third point if the response answers the basic elements of the user’s question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.\n- Grant a fourth point if the response is clearly written from an AI Assistant’s perspective, addressing the user’s question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.\n- Bestow a fifth point for a response that is impeccably tailored to the user’s question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.\n\nUser: {inst}\n\n<response>{resp}</response>\n\nAfter examining the user’s instruction and the response. Conclude with an integer score using the format: “Score: <total points>”\nRemember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary. To evaluate the response in alignment with this additive scoring model, we’ll systematically attribute points based on the outlined criteria."
                    target = f"Score: {score}"
                    
                    pairs.append(dict(
                        instruction=inst,
                        response=resp,
                        score=score,
                        raw_score=resp_qlt,
                        input=prompt,
                        output=target
                    ))
                            
        # allow only first turn
        if not first_turn:
            for reply in ins["replies"]:
                _traverse(reply)

    for tree in trees:
        prompt = tree["prompt"]
        _traverse(prompt)

    return pairs, resp_scores, converted_scores


if __name__ == "__main__":
    first_turn = True
    print(f"only allow first turn: {first_turn}")
    inst_qlt_thres = -1
    resp_qlt_thres = -1
    pairs, resp_scores, converted_scores = load_tree_data(
        "2023-04-12_oasst_ready.trees.jsonl", 
        inst_qlt_thres=inst_qlt_thres, 
        resp_qlt_thres=resp_qlt_thres, 
        first_turn=first_turn
    )

    print(f"Min: {min(resp_scores)}, {min(converted_scores)}")
    # create bar plot of response quality from range 0 to 1
    plt.hist(resp_scores, bins=5)
    plt.savefig("resp_qlt_thres.png")
    plt.clf()
    plt.hist(converted_scores, bins=5)
    plt.savefig("converted_scores.png")
    print(f"#data number: {len(pairs)}")

    # split 10% for validation
    random.shuffle(pairs)
    split = int(0.9 * len(pairs))
    train = pairs[:split]
    val = pairs[split:]

    train_data = {"type": "text2text", "instances": train}
    valid_data = {"type": "text2text", "instances": val}
    print(f"Train: {len(train_data['instances'])}, Valid: {len(valid_data['instances'])}")
    with open('eft_seed_train.json', 'w') as f:
        json.dump(train_data, f, indent=4)
    with open('eft_seed_valid.json', 'w') as f:
        json.dump(valid_data, f, indent=4)

