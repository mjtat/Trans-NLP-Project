#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:30:40 2017

@author: michelle
"""

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from string import punctuation
from gensim.models import Word2Vec, Phrases
from gensim import corpora, models

class TextAnalysis(object):
    
    # Initialize with a dataframe object
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def decode(self, column_title):
        dataframe = self.dataframe
        dataframe[column_title] = dataframe[column_title].astype(str)
        dataframe[column_title] = dataframe[column_title].str.lower()
        dataframe[column_title] = dataframe[column_title].str.decode('UTF-8', errors='strict')
        return dataframe
    
    # Takes an additional list of stop words to remove.    
    def defineStopwords(self, words = None):
        stop_words = stopwords.words('english')
        etc_stop = ['.', ',', '?', '!', '\'',  ':', '\"', '{', '}', ';', '%', '[',  ']', '(', ')', '-', '\'s', '\'ve', '...', '\'ll', '`', '``', '"n\'t"', '"\'m"', "''", '--', '&']
        if words is not None: 
            stop_words = stop_words + etc_stop + words
        
        else:
            stop_words = stop_words + etc_stop
            
        return stop_words
    
    def wordTokens(self, df_column, words = None, pos = False, encode = True):
        
        df = self.dataframe
        df = self.decode(df_column)
        corpus = df[df_column].tolist()

        
        cleaned_corpus = []
               
        stopwords = self.defineStopwords()
        
        for post in corpus:
            temp = post.lower()
            if encode == True:
                temp = post.decode('ascii', 'ignore')
                temp = post.encode('ascii', 'ignore')
            else:
                continue
            
            print temp
            temp = temp.translate(None, string.punctuation)
            tokenizer = RegexpTokenizer(r'\w+')
            temp = tokenizer.tokenize(temp)
            stopped_tokens = [i for i in temp if not i in stopwords]
            
            if pos == True:
                temp = nltk.pos_tag(temp)
            
            
            lemmatized_tokens = [nltk.WordNetLemmatizer().lemmatize(i) for i in stopped_tokens]
            cleaned_corpus.append(lemmatized_tokens)
            
        print '\n Successfully cleaned and tokenized abstracts.'
            
        return cleaned_corpus
            
        
        
    def sentenceTokens(self, df_column):
        df = self.dataframe
        df = self.decode(df_column)
        corpus = df[df_column].tolist()
        
        tokenized_sents = []
        cleaned_corpus = []
                
        for post in corpus:
            temp = post.lower()
            temp = post.encode('ascii', 'ignore')
            print temp
            #temp = str(TextBlob(temp).correct())
            temp = sent_tokenize(temp)             
            cleaned_corpus.append(temp)
        
        for sentence_list in cleaned_corpus:
            for sentence in sentence_list: 
                    temp = word_tokenize(sentence)
                    tokenized_sents.append(temp)
            
        return tokenized_sents
    
    def createBOW(self, corpus):
        dictionary = corpora.Dictionary(corpus)
        BOW = [dictionary.doc2bow(text) for text in corpus]
        tfidf = models.TfidfModel(BOW)
        corpus_tfidf = tfidf[BOW]
        return dictionary, corpus_tfidf
    
   
    def freqPlot(self, df):
        df = pd.DataFrame({'words': words, 'frequency': freq}) 
        fig, ax = plt.subplots()
        df.iloc[:,0:2].plot(kind = 'barh', figsize=(24,24), ax=ax, width = .8, fontsize = 20)
        ax.set_title("Word Pair Frequency", fontsize = 24)
        ax.set_yticklabels(df['words'])
        ax.set_xlabel("Frequency",fontsize = 24)
        ax.set_ylabel("Word Pairs", fontsize = 24)
        fig.tight_layout()
        plt.savefig('barplot_2.png', dpi = 300)
        
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
            

class DenseTransformer(object):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
            
if __name__ == '__main__':
    
    import os
    # Set working directory.
    os.chdir('/home/michelle/Documents/Blogs/Trans NLP/data')
    
    ## Determine relevant features for suicidality, using word2vec
    ask = pd.read_csv('allask.csv')  
    
    # Create an analysis object.
    trans_analysis = TextAnalysis(ask)
    
    # Tokenize sentences to do word2vec.
    trans_sents = trans_analysis.sentenceTokens('selftext')
    
    # Complete word2vec.
    sents = Word2Vec(trans_sents, size=200, window=5, min_count=10, workers=4, sg = 1)
    # Get the top 50 most associated features in the data.
    list_sim = sents.most_similar(positive = ['suicide', 'suicidal', 'empty'], negative = 'happy', topn=50)
    
    
 #################################################################################
 # From word associations, do some string searching to derive relevant posts for #
 # suicidality.                                                                  #
 #################################################################################
     word_list = []    
     for i in list_sim:
         word_list.append(i[0])
 
     # Do string matching to find relevant keywords. 
     # Use regex symbol | to specify "or"
    ask_distress = ask[ask['selftext'].str.contains('|'.join(word_list), na = False)]
    ask_distress = ask_distress[ask_distress['selftext'].str.len()>=1500]
    ask_distress = ask_distress.reset_index(level=0)
   
    # Set "at risk" posts as the label 1.
    ask_distress['distressed'] = 1
    
    # Find all other posts not matching keywords. Set those posts with label 0.
    ask_other = ask[~ask['selftext'].str.contains('|'.join(word_list), na = False)]
    ask_suicide = ask_other[ask_other['selftext'].str.len()>=1500]
    ask_other = ask_other.reset_index(level=0)
    ask_other = ask_other.sample(len(ask_suicide))
    ask_other['distressed'] = 0
    
    # Concatenate two dataframes together, generate test and training sets.
    frames = [ask_distress, ask_other]
    data = pd.concat(frames)
    data = data.reset_index(drop=True)
    train, test = train_test_split(data, test_size = 0.2, random_state = 42)

    # Generate y data.
    y_train = train['risk'].values
    y_test = test['risk'].values
    
    # Get word tokens for later analysis in scikit
    X_train_sents = TextAnalysis(train).sentenceTokens('selftext')
    X_test_sents = TextAnalysis(test).sentenceTokens('selftext')
    
    # Train word2vec model again to get Word Embedding Vectors
    model_train = Word2Vec(X_train_sents, size=200, window=10, min_count=10, workers=4, sg = 1)
    model_test = Word2Vec(X_test_sents, size=200, window=10, min_count=10, workers=4, sg = 1)
    
    # Average word vectors
    w2v_train = dict(zip(model_train.wv.index2word, model_train.wv.syn0))
    w2v_test = dict(zip(model_test.wv.index2word, model_test.wv.syn0))
    
    # Get word tokens for later scikit modeling.
    train, test = train_test_split(data, test_size = 0.2, random_state = 42)
    trans_tokens_train = TextAnalysis(train).wordTokens('selftext', encode = True)
    trans_tokens_test = TextAnalysis(test).wordTokens('selftext')
    
    # Generate BoW tfidf vectors.
    def SKBoW(tokens):
        corpus = []
        for doc in tokens:
            for word in doc:
                corpus.append(word)
        return corpus
    
    tf_train = SKBoW(trans_tokens_train)
    tf_test = SKBoW(trans_tokens_test)
    
    def SKtfidf(corpus, num_features):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(analyzer='word',
                        min_df = 0, 
                        stop_words = None, 
                        max_features = num_features)
        data = vectorizer.fit_transform(corpus)
        data = data.toarray(corpus)
        return data
    
    tf_train = SKtfidf(tf_train, 200)
    tf_test = SKtfidf(tf_test,200)    
    
    # Create a scikit pipeline for modeling word embeddings
    
    w2v_rf = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v_train)), 
                        ("Random Forest", RandomForestClassifier(n_estimators=500, criterion='gini'))])
                
    logistic = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v_train)), 
                        ("Logistic Reg", linear_model.LogisticRegression())])
    
    svc = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v_train)),
                   ("linear svc", svm.SVC(kernel="linear"))]) 
    
    gradient = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v_train)),
                   ("gradient boost", GradientBoostingClassifier(n_estimators = 500, learning_rate=1.0))]) 
    
    SDG = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v_train)),
                   ("SDG", SGDClassifier(loss="hinge", penalty="l2"))])  
    
    text_clf = Pipeline([('vect', CountVectorizer()),
...                      ('tfidf', TfidfTransformer()),
...                      ('clf', MultinomialNB()),
... ])
    

    rf_Bow = Pipeline([('tfidf', TfidfVectorizer(analyzer=lambda x: x,max_features = 200)), ('to dense', DenseTransformer()),
                        ("Random Forest", RandomForestClassifier(n_estimators=500, criterion='gini'))])
                
    logistic_BoW = Pipeline([('tfidf', TfidfVectorizer(analyzer=lambda x: x,max_features = 200)), ('to dense', DenseTransformer()), 
                        ("Logistic Reg", linear_model.LogisticRegression())])
    
    svc_BoW = Pipeline([('tfidf', TfidfVectorizer(analyzer=lambda x: x,max_features = 200)), ('to dense', DenseTransformer()),
                   ("linear svc", svm.SVC(kernel="linear"))]) 
    
    gradient_BoW = Pipeline([('tfidf', TfidfVectorizer(analyzer=lambda x: x,max_features = 200)), ('to dense', DenseTransformer()),
                   ("gradient boost", GradientBoostingClassifier(n_estimators = 500, learning_rate=1.0))]) 
    
    SDG_BoW = Pipeline([('tfidf', TfidfVectorizer(analyzer=lambda x: x,max_features = 200)), ('to dense', DenseTransformer()),
                   ("SDG", SGDClassifier(loss="hinge", penalty="l2"))])  

    


    all_models = [("random forest", w2v_rf),("logistic regression", logistic), ("linear svc", svc), ("gradient boosted trees", gradient), ("SDG-SVM", SDG)]
    all_models_BoW = [("random forest", rf_Bow),("logistic regression BoW", logistic_BoW), ("linear svc BoW", svc_BoW), ("gradient boosted trees", gradient_BoW), ("SDG-SVM", SDG_BoW)]
    
    scores = sorted([(name, cross_val_score(model, trans_tokens_train, y_train, cv=5).mean()) 
                     for name, model in all_models], 
                    key=lambda (_, x): -x)
    
    
    scores_BoW = sorted([(name, cross_val_score(model, trans_tokens_train, y_train, cv=5).mean()) 
                     for name, model in all_models_BoW], 
                    key=lambda (_, x): -x)
    
    print "\n" + tabulate(scores, floatfmt=".4f", headers=("model", 'score'))
    print "\n" + tabulate(scores_BoW, floatfmt=".4f", headers=("model", 'score'))
    
    plt.figure(figsize=(15, 8), dpi = 150)
    sns.set(font_scale=2)
    ax = sns.barplot(x=[name for name, _ in scores], y=[score for _, score in scores])
    ax.set_title("Model Classification Accuracy (KFold = 5)", fontsize = 24)
    ax.set_xlabel('Classification Model Name', fontsize = 24)
    ax.set_ylabel('Accuracy (Proportion Correct)', fontsize = 24)
    plt.ylim(.5,1)
    
    plt.figure(figsize=(17, 8), dpi = 150)
    sns.set(font_scale=2)
    ax = sns.barplot(x=[name for name, _ in scores_BoW], y=[score for _, score in scores_BoW])
    ax.set_title("Model Classification Accuracy (KFold = 5)", fontsize = 24)
    ax.set_xlabel('Classification Model Name', fontsize = 24)
    ax.set_ylabel('Accuracy (Proportion Correct)', fontsize = 24)
    plt.ylim(.5,1)
