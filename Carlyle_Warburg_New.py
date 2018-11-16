# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 15:39:33 2018

@author: lil
"""

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
#from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
import nltk
from nltk.util import ngrams # This is the ngram magic.
import os
from unidecode import unidecode

####################################################################################


Cfilename = 'carlyle_20180921.txt'
Wfilename = 'warburg_20180921.txt'
#####################################################################################

carlyledata = pd.read_csv(Cfilename, sep='\t', header=None, encoding='utf-8')
carlylelist = list(carlyledata[0])
warburgdata = pd.read_csv(Wfilename, sep='\t', header=None, encoding='utf-8')
warburglist = list(warburgdata[0])


query='''select distinct companyname from all_rating_event_vw'''
DBRS_Data = pd.read_sql(query, Oracle)
DBRS_Name = DBRS_Data['COMPANYNAME'].tolist()

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
    name = unidecode(name)
    a = name_breaker(name)
    for item in namelist:
        item = unidecode(item)
        item = item.lower().replace('.','')
        b = name_breaker(item)
        jd = jaccard_distance(name,item)
        cs = cosine_similarity_ngrams(a,b)
        total_score = jd + cs
        scoreList.append(total_score)
    max_score = max(scoreList)
    Index = scoreList.index(max(scoreList))
    best_match = namelist[Index]
    return best_match, max_score



output = []
print('%s carlyle names to match'%len(carlylelist))
for name in carlylelist:
    best_DBRS_Name, best_score = main(name, DBRS_Name)
    match = (name, best_DBRS_Name, best_score)
    output.append(match)
output_df = pd.DataFrame(output)


writer = pd.ExcelWriter('Carlyle_Matched.xlsx')
output_df.to_excel(writer, index=False)
writer.save()


output1 = []
print('%s warburg names to match'%len(warburglist))
for name in warburglist:
    best_DBRS_Name, best_score = main(name, DBRS_Name)
    match = (name, best_DBRS_Name, best_score)
    output1.append(match)
output_df1 = pd.DataFrame(output1)


writer = pd.ExcelWriter('Warburg_Matched.xlsx')
output_df1.to_excel(writer, index=False)
writer.save()



