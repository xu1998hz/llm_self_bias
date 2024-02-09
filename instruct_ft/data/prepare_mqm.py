import json
import csv
import re
import pandas as pd
from typing import Dict
from copy import deepcopy
import random

def parse_feedback(text):
    l = text.split('\n')
    out = []
    for line in l[1:]:
        try:
            match = re.search(r'Error Location \d: \'(.*)\', Error Type: (.*), Severity: (Major|Minor)', line)
            location = match.group(1)
            error_type = match.group(2).lower()
            severity = match.group(3).lower()
            s = f"'{location}' is a {severity} {error_type} error."
            out.append(s)
        except:
            return False
    return ' '.join(out)

def extract_context(text):
    # Regular expression to find all occurrences of <v>...</v>
    pattern = r"<v>(.*?)</v>"
    return re.findall(pattern, text)

def remove_tags(text):
    # Pattern to match <v> and </v> tags
    pattern = r"</?v>"
    return re.sub(pattern, "", text)

def read_tsv_and_convert_to_json(file_path, language):
    # Read the TSV file
    n = 10
    # df = pd.read_csv(file_path, delimiter='\t', encoding='utf-8', on_bad_lines='skip', nrows=5)
    # df = pd.read_csv(file_path, delimiter='\t', encoding='utf-8', nrows=n)

    # Convert DataFrame to a dictionary
    # data_dict = df.to_dict(orient='records')
    data_dict = []
    with open(file_path) as f:
        f.readline()
        for line in f:
            l = line.split('\t')
            d = dict(
                system=l[0],
                doc=l[1],
                doc_id=l[2],
                seg_id=l[3],
                rater=l[4],
                source=l[5],
                target=l[6],
                category=l[7],
                severity=l[8],
            )
            if language == 'zhen':
                d['severity'] = d['severity'][:-1]
            data_dict.append(d)

    # print(data_dict[:3])

    print(len(data_dict))

    # Initialize an empty dictionary for the JSON data
    json_data = {"type": "text2text"}
    json_data['data'] = {}
    json_data['instances'] = []

    # Process each row in the TSV file
    c = 0
    for row in data_dict:
        # Create the unique key
        unique_key = f"{row['system']}_{row['doc']}_{row['doc_id']}_{row['seg_id']}_{row['rater']}"

        # Check if the key is unique and add the data to the JSON dictionary
        if unique_key not in json_data['data']:
            if language == 'zhen':
                input = f"You are evaluating a Chinese-to-English Machine translation task. The source is '{remove_tags(row['source'])}'. The model generated translation is '{remove_tags(row['target'])}'. Please identify all errors in the translation, up to a maximum of five. For each error, please give me the corresponding error location, error type and major/minor label for each error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."
            if language == 'ende':
                input = f"You are evaluating a English-to-German Machine translation task. The source is '{remove_tags(row['source'])}'. The model generated translation is '{remove_tags(row['target'])}'. Please identify all errors in the translation, up to a maximum of five. For each error, please give me the corresponding error location, error type and major/minor label for each error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."
            json_data['data'][unique_key] = {
                'input': input,
                'output': '',
                'num': 0
            }

        if row['category'] == 'No-error':
            continue
        else:
            location = extract_context(row['target'])
            if len(location) > 1:
                # should not happen
                print(row)
            # if not in target, try to find in source (omit or source error)
            if len(location) == 0:
                location = extract_context(row['source'])
            if len(location) != 0 or row['category'] == 'Non-translation!':
                json_data['data'][unique_key]['num'] += 1
                n = json_data['data'][unique_key]['num']
                if row['category'] == 'Non-translation!':
                    s = f"\nError Location {n}: '{remove_tags(row['target'])}', Error Type: {row['category']}, Severity: {row['severity']}"
                else:
                    s = f"\nError Location {n}: '{location[0]}', Error Type: {row['category']}, Severity: {row['severity']}"
                json_data['data'][unique_key]['output'] += s
            else:
                # a couple bad datapoints here
                c += 1
        
    print(f'{c} bad datapoints with parsing error')
    # remove unique key and add count
    for k, v in json_data['data'].items():
        if v['num'] == 0:
            v['output'] = 'Your translation contains no errors.'
        else:
            rephrased = parse_feedback(v['output'])
            if rephrased == False:
                continue
            # v['output'] = f"Your translation contains {v['num']} errors." + rephrased
            v['output'] = rephrased
        del v['num']
        json_data['instances'].append(v)
    del json_data['data']
    # Return the JSON data
    return json_data

# Example usage
# file_path = '/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/data/mqm_newstest2021_zhen.tsv'
mode = 'zhen'
# mode = 'ende'
file_path = f'mqm_newstest2021_{mode}.tsv'
json_output = read_tsv_and_convert_to_json(file_path, mode)

# randomly remove 10% of the data as valid test set
n = len(json_output['instances'])
print(f'before: {n}')
random.shuffle(json_output['instances'])
n = int(n * 0.9)
valid_data = json_output['instances'][n:]
valid_data = {"type": "text2text", "instances": valid_data}
json_output['instances'] = json_output['instances'][:n]
print(f'after: {len(json_output["instances"])}')


# Optionally, write the JSON data to a file
# 'zhen' -> 'zh-en'
mode = mode[:2] + '-' + mode[2:]
with open(f'mqm_newstest2021_{mode}_train.json', 'w', encoding='utf-8') as json_file:
    json.dump(json_output, json_file, ensure_ascii=False, indent=4)
with open(f'mqm_newstest2021_{mode}_valid.json', 'w', encoding='utf-8') as json_file:
    json.dump(valid_data, json_file, ensure_ascii=False, indent=4)
