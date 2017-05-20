#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 20:43:18 2017

@author: michelle
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:30:40 2017

@author: michelle
"""
from sklearn.metrics import roc_curve, auc


import seaborn as sns
import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt
%matplotlib inline
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
from nltk import bigrams
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
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

        singletons = []
        bigrams = []
               
        stopwords = self.defineStopwords()
        
        for post in corpus:
            temp = post.lower()
            if encode == True:
                temp = post.encode('ascii', 'ignore').decode('ascii')
            else:
                continue
            
            print temp
            temp = temp.encode('utf-8').translate(None, string.punctuation)
            tokenizer = RegexpTokenizer(r'\w+')
            temp = tokenizer.tokenize(temp)
            stopped_tokens = [i for i in temp if not i in stopwords]
            
            
            if pos == True:
                temp = nltk.pos_tag(temp)
            
            
            lemmatized_tokens = [nltk.WordNetLemmatizer().lemmatize(i) for i in stopped_tokens]
            bigram = nltk.bigrams(lemmatized_tokens)
            bigram = list(bigram)
            singletons.append(lemmatized_tokens)
            bigrams.append(bigram)
            
        print '\n Successfully cleaned and tokenized abstracts.'
            
        return singletons, bigrams
            
        
        
    def sentenceTokens(self, df_column):
        df = self.dataframe
        df = self.decode(df_column)
        corpus = df[df_column].tolist()
        
        tokenized_sents = []
        cleaned_corpus = []
                
        for post in corpus:
            temp = post.lower()
            temp = post.encode('ascii', 'ignore').decode('ascii')
            print temp
            temp = sent_tokenize(temp)             
            cleaned_corpus.append(temp)
        
        for sentence_list in cleaned_corpus:
            for sentence in sentence_list: 
                    temp = word_tokenize(sentence)
                    tokenized_sents.append(temp)
            
        return tokenized_sents
      
if __name__ == '__main__':
    
    import os
    # Set working directory.
    os.chdir('/home/michelle/Documents/Blogs/Trans NLP/data/old')
    
    ## Determine relevant features for suicidality, using word2vec
    ask = pd.read_csv('allask.csv')  
    
    # Create an analysis object.
    trans_analysis = TextAnalysis(ask)

    # Clean and tokenize
    tokens, bgs = trans_analysis.wordTokens('selftext')

    # Create long corpus of tokens and bigrams
    token_corpus = []
    for doc in tokens:
        for token in doc:
            token_corpus.append(token)
        
    bigram_corpus = []
    for doc in bgs:
        for bigram in doc:
            bigram_corpus.append(bigram)
            
    fdist = nltk.FreqDist(token_corpus)
    fdist2 = nltk.FreqDist(bigram_corpus)
    
    words = []
    freq = []
    for word, frequency in fdist.most_common(75):
        words.append(word)
        freq.append(frequency)
        
    df = pd.DataFrame({'words': words, 'frequency': freq}) 
        
    fdist_bgs = nltk.FreqDist(bigram_corpus)
    
    def freqPlot(df):
        plt.figure(figsize=(12,24), dpi = 125)
        sns.set(font_scale=1.7)
        ax = sns.barplot(x=df['frequency'], y = df['words'], palette = 'colorblind')
        ax.set_title("Word Frequencies", fontsize = 20)
        ax.set_xlabel('Frequency (ocurrence in corpus)', fontsize = 20)
        ax.set_ylabel('Word / Word Pairing', fontsize = 20)



    bigram_list = []
    bgs_freq = []
    for word, frequency in fdist_bgs.most_common(100):
        bigram_list.append(word)
        bgs_freq.append(frequency)
    
    token_plot = trans_analysis.freqPlot(words, freq)
    bigram_plot = trans_analysis.freqPlot(bigram_list, bgs_freq)
    
    bigram_measures = BigramAssocMeasures()
    
    finder = BigramCollocationFinder.from_words(token_corpus)
    finder.apply_freq_filter(10)
    for i in finder.score_ngrams(bigram_measures.pmi):
        print i
    