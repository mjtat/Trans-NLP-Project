#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:30:40 2017

@author: michelle
"""

import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
from string import punctuation

class TextAnalysis(object):
    
    # Initialize with a dataframe object
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def decode(self, column_title):
        dataframe = self.dataframe
        dataframe[column_title] = dataframe[column_title].astype(str)
        dataframe[column_title] = dataframe[column_title].str.lower()
        dataframe[column_title] = dataframe[column_title].str.decode('UTF-8', errors='ignore')
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
    
    def tokenize_text(self, df_column, words = None):
        
        df = self.dataframe
        df = self.decode(df_column)
        corpus = df[df_column].tolist()
        
    
        cleaned_corpus = []
          
#        if words is not None:
#            stopwords = self.defineStopwords(words)
        
        stopwords = self.defineStopwords()
        
        for post in corpus:
            
            temp = post.lower()
            
            temp = post.encode('ascii', 'ignore')
            
            temp = temp.translate(None, string.punctuation)
            
            tokenizer = RegexpTokenizer(r'\w+')
                
#            tokenizer = TreebankWordTokenizer()
                
    #        temp = tokenizer.tokenize(temp)
            
            stopped_tokens = [i for i in temp if not i in stopwords]
            
            lemmatized_tokens = [nltk.WordNetLemmatizer().lemmatize(i) for i in stopped_tokens]
                
            cleaned_corpus.append(lemmatized_tokens)
            
        print '\n Successfully cleaned and tokenized abstracts.'
            
        return cleaned_corpus
    
    def sentence_tokens(self, df_column):
        df = self.dataframe
        df = self.decode(df_column)
        corpus = df[df_column].tolist()
        
        tokenized_sents = []
        cleaned_corpus = []
                
        for post in corpus:
            
            temp = post.lower()
            
            temp = post.encode('ascii', 'ignore')
            
            temp = sent_tokenize(temp) # I am happy, I am sad 
                      
            cleaned_corpus.append(temp)
            
        
        
        for sentence_list in cleaned_corpus:
            for sentence in sentence_list:
                temp = word_tokenize(sentence)
                tokenized_sents.append(temp)   
            
        return tokenized_sents
    

if __name__ == '__main__':
    import os
    # Set working directory.
    os.chdir('/home/michelle/Documents/Blogs/Trans NLP/data')
    
    ask = pd.read_csv('allask.csv')      

    frequency = TextAnalysis(ask)
    
    #frequency.decode('selftext')
    x = frequency.sentence_tokens('selftext')
    x = frequency.tokenize_text('selftext')
    
    tokenized_sents = []
    for sentence_list in x:
        for sentence in sentence_list:
            temp = word_tokenize(sentence)
            tokenized_sents.append(temp)
            
    model = Word2Vec(x, size=100, window=10, min_count=5, workers=4)
    model.most_similar(positive = ['tired', 'sleeping'], topn=20)