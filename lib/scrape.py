#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 20:30:06 2017

@author: michelle
"""
import os
# Set working directory.
os.chdir('/home/michelle/Documents/Blogs/Reddit')

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
from string import punctuation




def decode(dataframe,column_title):
    dataframe[column_title] = dataframe[column_title].astype(str)
    dataframe[column_title] = dataframe[column_title].str.lower()
    dataframe[column_title] = dataframe[column_title].str.decode('utf-8', errors='strict')
    return dataframe



def clean(corpus):
    cleaned_corpus = []
    stop_words = stopwords.words('english')
    etc_stop = ['.', ',', '?', '!', '\'',  ':', '\"', '{', '}', ';', '%', '[',  ']', '(', ')', '-', '\'s', '\'ve', '...', '\'ll', '`', '``', '"n\'t"', '"\'m"', "''", '--', '&']
    stop_words = stop_words + etc_stop 
    for post in corpus:
        temp = post.lower()
        temp = post.encode('ascii', 'ignore')
        temp = temp.translate(None, string.punctuation)
        tokenizer = RegexpTokenizer(r'\w+')
        temp = tokenizer.tokenize(temp)
        stopped_tokens = [i for i in temp if not i in stop_words]
        lemmatized_tokens = [nltk.WordNetLemmatizer().lemmatize(i) for i in stopped_tokens]
        cleaned_corpus.append(lemmatized_tokens)
    print '\n Successfully cleaned and tokenized abstracts.'
    return cleaned_corpus

if __name__ == '__main__':
    ask1 = pd.read_csv('ask1.csv')
    ask2 = pd.read_csv('ask2.csv')
    ask3 = pd.read_csv('ask3.csv')
    ask4 = pd.read_csv('ask4.csv')
    ask5 = pd.read_csv('ask5.csv')
    ask6 = pd.read_csv('ask6.csv')
    ask7 = pd.read_csv('ask7.csv')
    
    frames = [ask1, ask2, ask3, ask4, ask5, ask6, ask7]
    data = pd.concat(frames)
    
    data = decode(data, 'selftext')
    data = data['selftext'].tolist()
    
    data = clean(data)
    
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
