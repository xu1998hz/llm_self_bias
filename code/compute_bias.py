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

nlp = spacy.load("en_core_web_sm")
def detect_concepts(sentence: str, concepts: List[str]) -> Set[str]:
    present_concepts = []
    
    # Tokenize the sentence and lemmatize the tokens
    tokens = nltk.word_tokenize(sentence)
    lemmas = [token.lemma_ for token in nlp(sentence)]
    
    # Check if each concept is present in the sentence
    for concept in concepts:
        if concept in tokens or concept in lemmas:
            present_concepts.append(concept)
    
    return set(present_concepts)

@click.command()
@click.option("-bleurt_nor_file")
@click.option("-llm_score_file")
@click.option("-llm_out_file")
@click.option("-task_type")
def main(bleurt_nor_file, llm_score_file, task_type, llm_out_file):
    if task_type == 'commongen':
        src_lines = []
        with jsonlines.open('srcs/commongen_hard.jsonl') as reader:
            for line in reader:
                src_lines.append(line)
        src_lines = src_lines[:100]
        out_lines = open(llm_out_file, 'r')
        out_lines = [ele[:-1] for ele in out_lines]

    lines = open(
        llm_score_file,
        "r",
    ).readlines()
    final_ls = "".join(lines).split("[SEP_TOKEN_WENDA]")[:-1]
    score_ls = []
    for index, ele in enumerate(final_ls):
        if task_type == "mt":
            score_ls += [
                max(
                    -25,
                    (
                        -1 * ele.count("minor") + -5 * ele.count("major") + (-5) * ele.count("critical")
                    ),
                )
            ]
        else:
            total = len(src_lines[index]['concepts'])
            present_concepts = detect_concepts(out_lines[index], src_lines[index]['concepts'])
            gt_score = len(present_concepts)/total
            if gt_score < 1:
                gt_score = 0
            
            if ele[0] == '[' and ele[-1] == ']':
                pred_score = 0
            else:
                pred_score = 1
            print(pred_score-gt_score)

    # mapped_model = open(
    #     bleurt_nor_file, "r"
    # ).readlines()
    # mapped_model = [float(ele) for ele in mapped_model]
    # print("Model Score: ", sum(score_ls)/len(score_ls))
    # print("BLEURT Score: ", sum(mapped_model)/len(mapped_model))
    # # 1) Out of bia defination
    # diff_ls = [m_score - g_score for m_score, g_score in zip(score_ls, mapped_model)]
    # mean_bias = sum(diff_ls) / len(diff_ls)
    # print("Bias mean: ", mean_bias)

    # # 2) Use skewness to find the bias out of statistical distribution
    # skewsize = skewness(diff_ls)
    # print("Bias skewness: ", skewsize)

    # # 3) Study the shape of distribution
    # d_skew = distance_skewness(diff_ls, 0)
    # print("Bias distance skewness: ", d_skew)
    # print()


if __name__ == "__main__":
    main()
