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


#nltk.download('punkt')
#nltk.download('stopwords')


def tr_data(training_path):
	#Data will be a CSV or XML file
	try:
		general_tweets_corpus_train = pd.read_csv(training_path, encoding='utf-8')
	except:
		xml = objectify.parse(open(training_path))
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

	tweets_corpus = pd.concat([general_tweets_corpus_train])
	tweets_corpus = tweets_corpus.query('agreement != "DISAGREEMENT" and polarity != "NONE"')
	train_data = tweets_corpus[-tweets_corpus.content.str.contains('^http.*$')]
	return train_data

def ts_data(test_path):
	try:
		test_data = pd.read_csv(test_path, encoding='utf-8')
	except:
		xml = objectify.parse(open(test_path))
		#sample tweet object
		root = xml.getroot()
		test_data = pd.DataFrame(columns=('content', 'polarity'))
		tweets = root.getchildren()
		for i in range(0,len(tweets)):
		    tweet = tweets[i]
		    row = dict(zip(['content'], [tweet.content.text]))
		    row_s = pd.Series(row)
		    if row_s[0] != None:
		        row_s.name = i
		        test_data = test_data.append(row_s)


	test_data = pd.concat([test_data])
	
	

	return test_data


def tweets_classification(train_data, test_data, json=False, csv=True, result_path=None):
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

	#Stopwords: Empty word (i.e articles)

	spanish_stopwords = stopwords.words('spanish')
	stemmer = SnowballStemmer('spanish')


	#Non Words: Symbols and Numbers
	non_words = list(punctuation)
	non_words.extend(['¿', '¡'])
	non_words.extend(map(str,range(10)))


	#Binarizing

	train_data['polarity_bin'] = 0
	index = train_data.polarity.isin(['P', 'P+'])
	train_data.polarity_bin.loc[index] = 1


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

	best_pipe.fit(train_data.content, train_data.polarity_bin)


	test_data['polarity'] = best_pipe.predict(test_data.content)
	
	if csv:
		test_data.to_csv(result_path, encoding ='utf-8')
	elif json:
		test_data.to_json(result_path)


	return test_data
#tr, ts = load_data('TASS/csv/general-tweets-train-tagged.csv','Datasets/reformas/csv/RT.csv')

#result = tweets_classification(tr,ts, csv=False, json= True, result_path=0)
	











