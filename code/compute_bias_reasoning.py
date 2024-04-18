from datasets import load_dataset
import scipy.stats as stats
import numpy as np
from scipy import interpolate
from matplotlib import pyplot
import click
import jsonlines
from typing import Set, Dict, TypeVar, Iterable, List
import nltk
import spacy
import re
import pandas as pd
# we have three formula to compute bias


# This implementation is based on Wikipedia description
def distance_skewness(X, theta):
    # pairwise distances between the elements of X
    pairwise_distances = np.abs(np.subtract.outer(X, X))

    # numerator and denominator of the distance skewness formula
    numerator = np.sum(pairwise_distances)
    denominator = np.sum(np.abs(np.add.outer(X, X) - 2 * theta))

    # handle the case when Pr(X=theta) = 1
    if denominator == 0:
        return 0
    else:
        return 1 - numerator / denominator

def skewness(X):
    return sum([ele**3 for ele in X])/len(X) / (sum([ele**2 for ele in X])/len(X))**(3/2)

def extract_string(s):
    match = re.search(r"\\boxed\{(.*?)\}", s)
    if match:
        return match.group(1)

def get_max_key(d):
    return max(d, key=d.get)

def get_bias_self_consistency(file_name): # 'model_outputs/gpt-3.5-turbo/self_refine/math_20.txt'
    lines = open(file_name, 'r').readlines()
    lines = [ele[:-1] for ele in lines]
    cur_ans = [extract_string(ele.split('\t')[0]) for ele in lines]
    ans_lines = load_dataset('hendrycks/competition_math')['test']['solution'][:100]
    diff_ls = []

    for (i, k) in [(1, 11), (6, 16), (11, 21)]: #  
        count = 0
        temp_ls = []
        for line, ans, cand_ans in zip(lines, ans_lines, cur_ans):
            # cand_ans = extract_string(line.split('\t')[0])
            cur_dict = {}
            for other_ans in line.split('\t')[1:k]:
                if extract_string(other_ans) not in cur_dict:
                    cur_dict[extract_string(other_ans)]=0
                cur_dict[extract_string(other_ans)]+=1
        
            true_ans = extract_string(ans)
            if len(cur_dict) == 0:
                pred_ans="None"
            else:
                pred_ans = get_max_key(cur_dict)

            temp_ls+=[pred_ans]
            if true_ans == cand_ans:
                gt_ele=1
            else:
                gt_ele=0

            if pred_ans == cand_ans:
                pred_ele=1
            else:
                pred_ele=0

            if cand_ans == true_ans:
            # print(pred_ans, true_ans)
                count+=1
            # else:
            #     print(pred_ele-gt_ele)
            #print(pred_ele-gt_ele)
            diff_ls+=[pred_ele-gt_ele]
        cur_ans=temp_ls

        print(sum(diff_ls)/len(diff_ls))
        print(distance_skewness(diff_ls, 0))
        print(count)
        print()

def get_bias_reasoning(out_file_name, feedback_file_name):
    lines = open(out_file_name, 'r').readlines()
    lines = [ele[:-1] for ele in lines]
    f_lines = open(feedback_file_name, 'r').readlines()
    f_ls = []
    for ele in ''.join(f_lines).split('[SEP_TOKEN_WENDA]')[:-1]:
        if 'incorrect' in ele:
            f_ls+=[0]
        else:
            f_ls+=[1]

    cur_ans = [extract_string(ele.split('\t')[0]) for ele in lines]
    ans_lines = load_dataset('hendrycks/competition_math')['test']['solution'][:100]
    true_ans = [extract_string(ele) for ele in ans_lines]

    num_cor = 0
    total = 0
    diff_ls = []
    for cur, true, label in zip(cur_ans, true_ans, f_ls):
        if cur == true:
            num_cor+=1
            score = 1
        else:
            score = 0
        total+=1
        diff_ls += [label - score]
    print("Acc: ", num_cor/total)
    print("Bias: ", sum(diff_ls)/len(diff_ls))
    print("Dskew: ", distance_skewness(diff_ls, 0))

sys_name = "gemini"
for i in range(4):
    get_bias_reasoning(f'model_outputs/{sys_name}/self_refine/math_{i}.txt', f'model_outputs/{sys_name}/self_refine/math_{i}_feedback.txt')
    print()