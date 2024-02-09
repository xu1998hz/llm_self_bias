import json
import csv
import re
import pandas as pd
from typing import Dict
from copy import deepcopy
import random
# from mt_metrics_eval import data as mt_data

def parse_feedback(text, wmt):
    l = text.split('\n')
    out = []
    for line in l[1: 6]:
        try:
            if wmt == 'wmt21':
                match = re.search(r'Error Location \d: \'(.*)\', Error Type: (.*), Severity: (Major|Minor)', line)
            elif wmt == 'wmt22':
                match = re.search(r'Error Location \d: \'(.*)\', Error Type: (.*), Severity: (major|minor)', line)
            location = match.group(1)
            error_type = match.group(2).lower()
            severity = match.group(3).lower()
            s = f"'{location}' is a {severity} {error_type} error."
            out.append(s)
        except:
            return False
    assert len(out) <= 5
    return ' '.join(out)

def extract_context(text):
    # Regular expression to find all occurrences of <v>...</v>
    pattern = r"<v>(.*?)</v>"
    return re.findall(pattern, text)

def remove_tags(text):
    # Pattern to match <v> and </v> tags
    pattern = r"</?v>"
    return re.sub(pattern, "", text)

def read_tsv_and_convert_to_json(file_path, srcs, refs, language, wmt):
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

    print(len(data_dict))

    # Initialize an empty dictionary for the JSON data
    json_data = {"type": "text2text"}
    json_data['data'] = {}
    json_data['instances'] = []
    src_dict = {}

    if language == 'zhen':
        src_lang, tgt_lang = 'Chinese', 'English'
    elif language == 'ende':
        src_lang, tgt_lang = 'English', 'German'

    # Process each row in the TSV file
    c = 0
    p = 0
    b = 0
    for i, row in enumerate(data_dict):
        if row['system'] in ['refA.en', 'ref.A']:
                ref_key = (row['doc'], row['doc_id'], row['seg_id'])
                src_dict[ref_key] = remove_tags(row['target'])
                continue

        # Create the unique key
        unique_key = (row['system'], row['doc'], row['doc_id'], row['seg_id'], row['rater'])

        # Check if the key is unique and add the data to the JSON dictionary
        if unique_key not in json_data['data']:
            json_data['data'][unique_key] = {
                'mt': remove_tags(row['target']),
                'output': '',
                'num': 0
            }

        if row['category'] == 'No-error':
            continue
        else:
            location = extract_context(row['target'])
            # if not in target, error appears in source and skip in this case
            if len(location) == 0 and row['category'] != 'Non-translation!':
                continue
            else:
                json_data['data'][unique_key]['num'] += 1
                n = json_data['data'][unique_key]['num']
                if n > 5:
                    b += 1
                if row['category'] == 'Non-translation!':
                    s = f"\nError Location {n}: '{remove_tags(row['target'])}', Error Type: {row['category']}, Severity: {row['severity']}"
                else:
                    s = f"\nError Location {n}: '{location[0]}', Error Type: {row['category']}, Severity: {row['severity']}"
                json_data['data'][unique_key]['output'] += s
        
    print('size of src dict = ', len(src_dict.keys()))

    # postprocess
    for k, v in json_data['data'].items():
        if v['num'] == 0:
            # continue
            v['output'] = 'Your translation contains no errors.'
            p += 1
        else:
            rephrased = parse_feedback(v['output'], wmt)
            if rephrased == False:
                continue
            v['output'] = rephrased
        del v['num']

        ref = src_dict.get(k[1: -1], None)
        if ref == None:
            c += 1
            continue
        template = f"You are evaluating a Machine translation task. The reference translation is '{ref}'. The model generated translation is '{v['mt']}'. Please identify all errors in the translation, up to a maximum of five. For each error, please give me the corresponding error location, error type and major/minor label for each error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."
        v['input'] = template
        del v['mt']

        json_data['instances'].append(v)

    del json_data['data']

    print('no ref: ', c)
    print('no error =', p)
    print('more than five error: ', b)

    return json_data

# Example usage
mode = 'zhen' # or 'ende'
wmt = 'wmt21' # or 'wmt22'
if wmt == 'wmt21':
    file_path = f'mqm_newstest2021_{mode}.tsv'
else:
    file_path = f'mqm_generalMT2022_{mode}.tsv'

srcs, refs = None, None # no mt data on gemini for wenda's account, ignore for now

json_output = read_tsv_and_convert_to_json(file_path, srcs, refs, mode, wmt)

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
with open(f'{wmt}_{mode}_ref_train.json', 'w', encoding='utf-8') as json_file:
    json.dump(json_output, json_file, ensure_ascii=False, indent=4)
with open(f'{wmt}_{mode}_ref_valid.json', 'w', encoding='utf-8') as json_file:
    json.dump(valid_data, json_file, ensure_ascii=False, indent=4)
