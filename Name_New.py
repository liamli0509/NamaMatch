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

####################################################################################
Oracle = cx_Oracle.connect('unidb/fnRjn5LLuuZLAJLQLuQr@amz-p-ora20.dbrs.local:1521/BRAPP')
os.chdir('U:/ECB/Coverage')
#####################################################################################

ECB_Rated = pd.read_csv('ea_csv_180723_rated.csv',encoding='utf-8', sep='\t')
ECB_Name = ECB_Rated['ISSUER_NAME'].tolist()
ECB_Name = list(set(ECB_Name))

query='''select COMPANY_ID, NAME from COMPANY where IS_ACTIVE='Y' order by COMPANY_ID'''
DBRS_Data = pd.read_sql(query, Oracle)
DBRS_Name = DBRS_Data['NAME'].tolist()

#ECB = pd.read_csv('ECB.txt',encoding='utf-8', sep='\t')
#ECB_Name = ECB['ECB_ISSUER_NAME'].tolist()
#DBRS = pd.read_csv('DBRS.txt',encoding='utf-8', sep='\t')
#DBRS_Name = DBRS['DBRS_ISSUER_NAME'].tolist()



NGRAM = 1

re_sent_ends_naive = re.compile(r'[.\n]')
re_stripper_alpha = re.compile('[^a-zA-Z0-9]+')
re_stripper_naive = re.compile('[^a-zA-Z0-9\.\n]')

splitter_naive = lambda x: re_sent_ends_naive.split(re_stripper_naive.sub(' ', x))

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

def name_breaker(txt):
    """Get tuples that ignores all punctuation (including sentences)."""
    if not txt: return None
    ng = ngrams(re_stripper_alpha.sub(' ', txt).split(), NGRAM)
    return list(ng)


def jaccard_distance(a, b):
    """Calculate the jaccard distance between sets A and B"""
    a = set(a)
    b = set(b)
    return 1.0 * len(a&b)/len(a|b)

def cosine_similarity_ngrams(a, b):
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

def main(name, namelist):
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
print('%s names to match'%len(ECB_Name))
for name in ECB_Name:
    best_DBRS_Name, best_score = main(name, DBRS_Name)
    match = (name, best_DBRS_Name, best_score)
    output.append(match)
output_df = pd.DataFrame(output)


writer = pd.ExcelWriter('ECB_Matched.xlsx')
output_df.to_excel(writer, index=False)
writer.save()