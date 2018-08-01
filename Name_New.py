# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:23:58 2018

@author: lil
"""


import re
import math
import pandas as pd
from datetime import date, timedelta
import cx_Oracle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from itertools import chain
from collections import Counter
import nltk
from nltk.util import ngrams # This is the ngram magic.
import os


#####################################################################################

DataSet1 = pd.read_csv(filename1,encoding='utf-8', sep='\t')
Name1 = DataSet1['ISSUER_NAME'].tolist()
Name1 = list(set(ECB_Name))

DataSet2 = pd.read_csv(filename2,encoding='utf-8', sep='\t')
Name2 = DataSet2['ISSUER_NAME'].tolist()
Name2 = list(set(ECB_Name))





NGRAM = 1

re_sent_ends_naive = re.compile(r'[.\n]')
re_stripper_alpha = re.compile('[^a-zA-Z0-9]+')
re_stripper_naive = re.compile('[^a-zA-Z0-9\.\n]')

splitter_naive = lambda x: re_sent_ends_naive.split(re_stripper_naive.sub(' ', x))

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

def name_breaker(txt):#break down name by ngrams
    if not txt: return None
    ng = ngrams(re_stripper_alpha.sub(' ', txt).split(), NGRAM)
    return list(ng)


def jaccard_distance(a, b):#compute the jaccard distance between two names
    a = set(a)
    b = set(b)
    return 1.0 * len(a&b)/len(a|b)

def cosine_similarity_ngrams(a, b):#compute the cosine similarity between two names
    vec1 = Counter(a)
    vec2 = Counter(b)
    
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator) / denominator

def main(name, namelist):#compare name from dataset1 to all names in dataset2, get the best match
    scoreList = []
    name = name.lower().replace('.','')
    a = name_breaker(name)
    for whatever in namelist:
        whatever = whatever.lower().replace('.','')
        b = name_breaker(whatever)
        jd = jaccard_distance(a,b)
        cs = cosine_similarity_ngrams(a,b)
        total_score = jd + cs
        scoreList.append(total_score)
    max_score = max(scoreList)
    Index = scoreList.index(max(scoreList))
    best_match = namelist[Index]
    return best_match, max_score



output = []
print('%s names to match'%len(Name1))
for name in Name1:
    best_DBRS_Name, best_score = main(name, Name2)
    match = (name, best_DBRS_Name, best_score)
    output.append(match)
output_df = pd.DataFrame(output)


writer = pd.ExcelWriter('Name_Matched.xlsx')
output_df.to_excel(writer, index=False)
writer.save()
