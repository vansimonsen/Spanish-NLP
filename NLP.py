#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import unicodedata
import re
import pandas as pd
pd.set_option('max_colwidth',1000)
from lxml import objectify
import numpy as np

from xml.etree.ElementTree import Element, SubElement, Comment, tostring, ElementTree


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
    stemmer = SnowballStemmer('spanish')

    non_words = list(punctuation)
    non_words.extend(['¿', '¡'])
    non_words.extend(map(str,range(10)))

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

def anotherDataXML(origin_path, output_path):
    data = pd.read_csv(origin_path, encoding='utf-8')

    tweets = Element('tweets')

    

    for i in data['content']:

        tweet = SubElement(tweets, 'tweet')

        tweet_id = SubElement(tweet, 'tweetid')
        tweet_id.text = ''

        user = SubElement(tweet, 'user')
        user.text = ''

        content = SubElement(tweet, 'content')
        content.text = i


        date = SubElement(tweet, 'date')
        date.text = ''

        lang = SubElement(tweet, 'lang')
        lang.text = 'es'

        sentiments = SubElement(tweet, 'sentiments')
        
        polarity = SubElement(sentiments, 'polarity')
        
        value = SubElement(polarity, 'value')
        value.text = ''
  
    tree = ElementTree(tweets)
    tree.write(output_path)


def read_data(train_path, test_path, save=False, save_path=None):
    try:
        train_data = pd.read_csv(train_path, encoding='utf-8')
    except:
        xml = objectify.parse(open(train_path))
        #sample tweet object
        root = xml.getroot()
        train_data = pd.DataFrame(columns=('content', 'polarity', 'agreement'))
        tweets = root.getchildren()
        for i in range(0,len(tweets)):
            tweet = tweets[i]
            row = dict(zip(['content', 'polarity', 'agreement'], 
                           [tweet.content.text, tweet.sentiments.polarity.value.text, 
                            tweet.sentiments.polarity.type.text]))
            row_s = pd.Series(row)
            row_s.name = i
            train_data = train_data.append(row_s)


    try:
        test = pd.read_csv(test_path, encoding='utf-8')
    except:
        xml = objectify.parse(open(test_path))
        #sample tweet object
        root = xml.getroot()
        test = pd.DataFrame(columns=('content', 'polarity'))
        tweets = root.getchildren()
        for i in range(0,len(tweets)):
            tweet = tweets[i]
            row = dict(zip(['content'], [tweet.content.text]))
            row_s = pd.Series(row)
            if row_s[0] != None:
                row_s.name = i
                test = test.append(row_s)
        if save:
            test.to_csv(save_path, index=False, encoding='utf-8')

    return train_data, test


def nlp(train, test):
    test = pd.concat([test])

    tweets_corpus = pd.concat([train])
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


    test['polarity'] = best_pipe.predict(test.content)
    test.to_csv('rt_predicted.csv', encoding ='utf-8')
    return test

#tr_re,ts_re = read_data('TASS/csv/general-tweets-train-tagged.csv', 're-test.csv')
#tr,ts = read_data('TASS/csv/general-tweets-train-tagged.csv', 'rc-test.csv')
#tr,ts = read_data('TASS/csv/general-tweets-train-tagged.csv', 'rl-test.csv')
tr,ts = read_data('TASS/csv/general-tweets-train-tagged.csv', 'rt-test.csv')

nlp(tr,ts)
    