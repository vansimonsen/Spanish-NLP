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


#Required packages from nltk
#nltk.download('punkt')
#nltk.download('stopwords')



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


tweets_corpus = pd.concat([general_tweets_corpus_train])
tweets_corpus = tweets_corpus.query('agreement != "DISAGREEMENT" and polarity != "NONE"')
tweets_corpus = tweets_corpus[-tweets_corpus.content.str.contains('^http.*$')]



try:
    general_tweets_corpus_test = pd.read_csv('datasets/csv/general-tweets-test1k.csv')#, encoding='utf-8')
except:
    xml = objectify.parse(open('datasets/xml/general-tweets-test1k.xml'))
    #sample tweet object
    root = xml.getroot()
    general_tweets_corpus_test = pd.DataFrame(columns=('content', 'polarity'))
    tweets = root.getchildren()
    for i in range(0,len(tweets)):
        tweet = tweets[i]
        row = dict(zip(['content'], [tweet.content.text]))
        row_s = pd.Series(row)
        row_s.name = i
        general_tweets_corpus_test = general_tweets_corpus_test.append(row_s)
    general_tweets_corpus_test.to_csv('datasets/csv/general-tweets-test1k.csv', index=False)#, encoding='utf-8')


tweets_test = pd.concat([general_tweets_corpus_test])




try:
    tagged_tweets_corpus_test = pd.read_csv('datasets/csv/general-tweets-test1k-tagged.csv', encoding='utf-8')
except:

    from lxml import objectify
    xml = objectify.parse(open('datasets/xml/general-tweets-test1k-tagged.xml'))
    #sample tweet object
    root = xml.getroot()
    tagged_tweets_corpus_test = pd.DataFrame(columns=('content', 'polarity'))
    tweets = root.getchildren()
    for i in range(0,len(tweets)):
        tweet = tweets[i]
        row = dict(zip(['content', 'polarity', 'agreement'], [tweet.content.text, tweet.sentiments.polarity.value.text]))
        row_s = pd.Series(row)
        row_s.name = i
        tagged_tweets_corpus_test = tagged_tweets_corpus_test.append(row_s)
    tagged_tweets_corpus_test.to_csv('datasets/csv/general-tweets-test1k-tagged.csv', index=False, encoding='utf-8')



tweets_tagged = pd.concat([tagged_tweets_corpus_test])
tweets_tagged = tweets_tagged.query('polarity != "NONE"')
diff = np.setdiff1d(tweets_test.index.values, tweets_tagged.index.values)

tweets_test = tweets_test.drop(diff)


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
print tweets_corpus.polarity_bin.value_counts(normalize=True)

tweets_test['polarity_bin'] = 0

tweets_tagged['polarity_bin'] = 0
index = tweets_tagged.polarity.isin(['P', 'P+'])
tweets_tagged.polarity_bin.loc[index] = 1
tweets_tagged.polarity_bin.value_counts(normalize=True)

y = tweets_tagged.polarity_bin.values


vectorizer = CountVectorizer(
                analyzer = 'word',
                tokenizer = tokenize,
                lowercase = True,
                stop_words = spanish_stopwords)

pipeline = Pipeline([
    ('vect', vectorizer),
    ('cls', LinearSVC()),
])



parameters = {
    'vect__max_df': (0.5, 1.9),
    'vect__min_df': (10, 20,50),
    'vect__max_features': (500, 1000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'cls__C': (0.2, 0.5, 0.7),
    'cls__loss': ('hinge', 'squared_hinge'),
    'cls__max_iter': (500, 1000)
}


grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1 , scoring='roc_auc')
grid_search.fit(tweets_corpus.content, tweets_corpus.polarity_bin)


print grid_search.best_params_

from sklearn.externals import joblib
joblib.dump(grid_search, 'grid_search.pkl')


model = LinearSVC(C=.2, loss='squared_hinge',max_iter=1000,multi_class='ovr',
              random_state=None,
              penalty='l2',
              tol=0.0001
)

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = spanish_stopwords,
    min_df = 50,
    max_df = 1.9,
    ngram_range=(1, 1),
    max_features=1000
)

corpus_data_features = vectorizer.fit_transform(tweets_corpus.content)
corpus_data_features_nd = corpus_data_features.toarray()


scores = cross_val_score(
    model,
    corpus_data_features_nd[0:len(tweets_corpus)],
    y=tweets_corpus.polarity_bin,
    scoring='roc_auc',
    cv=5
    )

print scores.mean()


##RESULTS

#{'vect__ngram_range': (1, 2), 'cls__loss': 'hinge', 
# 'vect__max_df': 0.5, 'cls__max_iter': 1000, 'vect__min_df': 10, 
# 'vect__max_features': 1000, 'cls__C': 0.2}

#0.778635069135




