import json
import random
f = 'seed_first_turn.jsonl'

# read data in jsonl format
with open(f, 'r') as file:
    data = file.readlines()
    data = [json.loads(line) for line in data]

data_dict = []

print(len(data))
for i in range(len(data)):
    d = {}
    d['input'] = data[i]['instruction']
    d['output'] = data[i]['response']
    d['instruction_quality'] = data[i]['instruction_quality']
    d['response_quality'] = data[i]['response_quality']
    data_dict.append(d)

json_data = {"type": "text2text"}
json_data['instances'] = data_dict

with open('ift_seed.json', 'w') as file:
    json.dump(json_data, file, indent=4)