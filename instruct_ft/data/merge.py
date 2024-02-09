import json
mqm = 'mqm_newstest2021_zh-en_train.json'
ift = 'ift_seed.json'

# merge two json
with open(mqm, 'r') as f1:
    mqm = json.load(f1)
with open(ift, 'r') as f2:
    ift = json.load(f2)
new_data = {"type": "text2text", "instances": mqm["instances"] + ift["instances"]}
with open('ift_mqm.json', 'w') as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)