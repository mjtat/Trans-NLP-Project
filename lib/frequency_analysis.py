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
from string import punctuation

class frequency_analysis(object):
    
    # Initialize with a dataframe object
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def decode(self, dataframe,column_title):
        dataframe[column_title] = dataframe[column_title].astype(str)
        dataframe[column_title] = dataframe[column_title].str.lower()
        dataframe[column_title] = dataframe[column_title].str.decode('utf-8', errors='strict')
        return dataframe
    
    # Takes an additional list of stop words to remove.    
    def defineStopwords(self, words = None):
        stop_words = stopwords.words('english')
        etc_stop = ['.', ',', '?', '!', '\'',  ':', '\"', '{', '}', ';', '%', '[',  ']', '(', ')', '-', '\'s', '\'ve', '...', '\'ll', '`', '``', '"n\'t"', '"\'m"', "''", '--', '&']
        stop_words = stop_words + etc_stop + self.words
        return stop_words
    
    def tokenize(self, strip_punct = True, regexp = True, treebank = False, lemmatize = True):
        
        cleaned_corpus = []
                   
        for post in corpus:
            temp = post.lower()
            
            temp = post.encode('ascii', 'ignore')
            
            if self.strip_punct == True:
                temp = temp.translate(None, string.punctuation)
            
            if self.regexp == True and self.treebank == False:
                tokenizer = RegexpTokenizer(r'\w+')
                
            elif self.treebank == True and self.regexp == False:
                tokenizer = TreebankWordTokenizer()
                
                
            temp = tokenizer.tokenize(temp)
            
            stopped_tokens = [i for i in temp if not i in defineStopwords()]
            
            if self.lemmatize == True:
                lemmatized_tokens = [nltk.WordNetLemmatizer().lemmatize(i) for i in stopped_tokens]
            
            cleaned_corpus.append(lemmatized_tokens)
            print '\n Successfully cleaned and tokenized abstracts.'
            
        return cleaned_corpus
        