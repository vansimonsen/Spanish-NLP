#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import unicodedata
import re
import pandas as pd
pd.set_option('max_colwidth',1000)
from lxml import objectify
import numpy as np



from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer

from string import punctuation

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer    


#Stem: Cut word in root (wait: wait, waited: wait, waiting: wait)
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

#Each word is a token
def tokenize(text):
    text = ''.join([c for c in text if c not in non_words])
    tokens =  word_tokenize(text)

    # stem
    try:
        stems = stem_tokens(tokens, stemmer)
    except Exception as e:
        print(e)
        print(text)
        stems = ['']
    return stems

try:
    general_tweets_corpus_train = pd.read_csv('datasets/csv/general-tweets-train-tagged.csv', encoding='utf-8')
except:
    xml = objectify.parse(open('datasets/xml/general-tweets-train-tagged.xml'))
    #sample tweet object
    root = xml.getroot()
    general_tweets_corpus_train = pd.DataFrame(columns=('content', 'polarity', 'agreement'))
    tweets = root.getchildren()
    for i in range(0,len(tweets)):
        tweet = tweets[i]
        row = dict(zip(['content', 'polarity', 'agreement'], 
                       [tweet.content.text, tweet.sentiments.polarity.value.text, 
                        tweet.sentiments.polarity.type.text]))
        row_s = pd.Series(row)
        row_s.name = i
        general_tweets_corpus_train = general_tweets_corpus_train.append(row_s)
    general_tweets_corpus_train.to_csv('datasets/csv/general-tweets-train-tagged.csv', index=False, encoding='utf-8')



try:
    re_test = pd.read_csv('reformas/csv/RE.csv', encoding='utf-8')
except:
    xml = objectify.parse(open('reformas/xml/RE.xml'))
    #sample tweet object
    root = xml.getroot()
    re_test = pd.DataFrame(columns=('content', 'polarity'))
    tweets = root.getchildren()
    for i in range(0,len(tweets)):
        tweet = tweets[i]
        row = dict(zip(['content'], [tweet.content.text]))
        row_s = pd.Series(row)
        if row_s[0] != None:
            row_s.name = i
            re_test = re_test.append(row_s)
    re_test.to_csv('reformas/csv/RE.csv', index=False, encoding='utf-8')


try:
    rc_test = pd.read_csv('reformas/csv/RC.csv', encoding='utf-8')
except:
    xml = objectify.parse(open('reformas/xml/RC.xml'))
    #sample tweet object
    root = xml.getroot()
    rc_test = pd.DataFrame(columns=('content', 'polarity'))
    tweets = root.getchildren()
    for i in range(0,len(tweets)):
        tweet = tweets[i]
        row = dict(zip(['content'], [tweet.content.text]))
        row_s = pd.Series(row)
        if row_s[0] != None:
            row_s.name = i
            rc_test = rc_test.append(row_s)
    rc_test.to_csv('reformas/csv/RC.csv', index=False, encoding='utf-8')

try:
    rt_test = pd.read_csv('reformas/csv/RT.csv', encoding='utf-8')
except:
    xml = objectify.parse(open('reformas/xml/RT.xml'))
    #sample tweet object
    root = xml.getroot()
    rt_test = pd.DataFrame(columns=('content', 'polarity'))
    tweets = root.getchildren()
    for i in range(0,len(tweets)):
        tweet = tweets[i]
        row = dict(zip(['content'], [tweet.content.text]))
        row_s = pd.Series(row)
        if row_s[0] != None:
            row_s.name = i
            rt_test = rt_test.append(row_s)
    rt_test.to_csv('reformas/csv/RT.csv', index=False, encoding='utf-8')

try:
    rl_test = pd.read_csv('reformas/csv/RL.csv', encoding='utf-8')
except:
    xml = objectify.parse(open('reformas/xml/RL.xml'))
    #sample tweet object
    root = xml.getroot()
    rl_test = pd.DataFrame(columns=('content', 'polarity'))
    tweets = root.getchildren()
    for i in range(0,len(tweets)):
        tweet = tweets[i]
        row = dict(zip(['content'], [tweet.content.text]))
        row_s = pd.Series(row)
        if row_s[0] != None:
            row_s.name = i
            rl_test = rl_test.append(row_s)
    rl_test.to_csv('reformas/csv/RL.csv', index=False, encoding='utf-8')


re_test = pd.concat([re_test])
rc_test = pd.concat([rc_test])
rl_test = pd.concat([rl_test])
rt_test = pd.concat([rt_test])

tweets_corpus = pd.concat([general_tweets_corpus_train])
tweets_corpus = tweets_corpus.query('agreement != "DISAGREEMENT" and polarity != "NONE"')
tweets_corpus = tweets_corpus[-tweets_corpus.content.str.contains('^http.*$')]

#Stopwords: Empty word (i.e articles)

spanish_stopwords = stopwords.words('spanish')
stemmer = SnowballStemmer('spanish')


#Non Words: Symbols and Numbers
non_words = list(punctuation)
non_words.extend(['¿', '¡'])
non_words.extend(map(str,range(10)))


#Binarizing

tweets_corpus['polarity_bin'] = 0
index = tweets_corpus.polarity.isin(['P', 'P+'])
tweets_corpus.polarity_bin.loc[index] = 1


### BEST PARAMS

best_params = {'vect__ngram_range': (1, 2), 'cls__loss': 'hinge', 'vect__max_df': 0.5
 , 'cls__max_iter': 1000, 'vect__min_df': 10, 'vect__max_features': 1000
 , 'cls__C': 0.2}

best_pipe = Pipeline([
    ('vect', CountVectorizer(
            analyzer = 'word',
            tokenizer = tokenize,
            lowercase = True,
            stop_words = spanish_stopwords,
            min_df = 10,
            max_df = 0.5,
            ngram_range=(1, 2),
            max_features=1000
            )),
    ('cls', LinearSVC(C=.2, loss='hinge',max_iter=1000,multi_class='ovr',
             random_state=None,
             penalty='l2',
             tol=0.0001
             )),
])

best_pipe.fit(tweets_corpus.content, tweets_corpus.polarity_bin)


re_test['polarity'] = best_pipe.predict(re_test.content)
rc_test['polarity'] = best_pipe.predict(rc_test.content)
rl_test['polarity'] = best_pipe.predict(rl_test.content)
rt_test['polarity'] = best_pipe.predict(rt_test.content)

re_test.to_csv('re_predicted.csv', encoding ='utf-8')
rc_test.to_csv('rc_predicted.csv', encoding ='utf-8')
rl_test.to_csv('rl_predicted.csv', encoding ='utf-8')
rt_test.to_csv('rt_predicted.csv', encoding ='utf-8')



