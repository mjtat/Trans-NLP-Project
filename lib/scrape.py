#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 20:30:06 2017

@author: michelle
"""
import pandas as pd
import praw
import nltk
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import StanfordTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import word_tokenize
from string import punctuation


class FreqAnalysis(object):
    
    def __init__(self, df):
        self.df = df

    def decode(self, column_title):
        dataframe = self.df
        dataframe[column_title] = dataframe[column_title].astype(str)
        dataframe[column_title] = dataframe[column_title].str.lower()
        dataframe[column_title] = dataframe[column_title].str.decode('utf-8', errors='strict')
        corpus = data[column_title].tolist()
        return corpus

    def clean(self, column_title, strip_punct = True, regexp = True):
        corpus = self.decode(column_title)
        cleaned_corpus = []
        stop_words = stopwords.words('english')
        etc_stop = ['.', ',', '?', '!', '\'',  ':', '\"', '{', '}', ';', '%', '[',  ']', '(', ')', '-', '\'s', '\'ve', '...', '\'ll', '`', '``', '"n\'t"', '"\'m"', "''", '--', '&']
        stop_words = stop_words + etc_stop 
        for post in corpus:
            temp = post.lower()
            temp = post.encode('ascii', 'ignore')
            
            if strip_punct == True:
                temp = temp.translate(None, string.punctuation)
                        
            if regexp == True:
                tokenizer = RegexpTokenizer(r'\w+')
                temp = tokenizer.tokenize(temp)
            else:
                temp = word_tokenize(temp)
                
            
            
            stopped_tokens = [i for i in temp if not i in stop_words]
            lemmatized_tokens = [nltk.WordNetLemmatizer().lemmatize(i) for i in stopped_tokens]
            cleaned_corpus.append(lemmatized_tokens)
        print '\n Successfully cleaned and tokenized abstracts.'
        return cleaned_corpus
    
    def count_ngram(self, corpus, ngram, top):
        word_list = []
        words = []
        freq = []
        
        for i in corpus:
            word_list += i
            
        fdist = FreqDist(list(nltk.ngrams(word_list, ngram)))
        
        for word, frequency in fdist.most_common(top):
            print word, frequency
            
            words.append(word)
            freq.append(frequency)
    
        df = pd.DataFrame({'words': words, 'frequency': freq})
        
        return df
            
if __name__ == '__main__':
    data = pd.read_csv('allask.csv')
    
    test = FreqAnalysis(data)
    x = test.clean('selftext')

    y = test.count_ngram(x, 2, 50)    
   
    word_list = []
    for i in data:
        word_list += i
        
    fdist = FreqDist(list(nltk.ngrams(word_list,2)))
    
    words = []
    freq = []
    
    for word, frequency in fdist.most_common(75):
        words.append(word)
        freq.append(frequency)
    
    df = pd.DataFrame({'words': words, 'frequency': freq})
    
    
    fig, ax = plt.subplots()
    df.iloc[:,0:2].plot(kind = 'barh', figsize=(24,24), ax=ax, width = .8, fontsize = 20)
    ax.set_title("Word Pair Frequency", fontsize = 24)
    ax.set_yticklabels(df['words'])
    ax.set_xlabel("Frequency",fontsize = 24)
    ax.set_ylabel("Word Pairs", fontsize = 24)
    fig.tight_layout()
    plt.savefig('barplot_2.png', dpi = 300)
