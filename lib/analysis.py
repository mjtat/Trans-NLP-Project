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
from nltk import ngrams
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
            temp = post.encode('ascii', 'ignore').decode('ascii')
            print temp
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
    y_train = train['distressed'].values
    y_test = test['distressed'].values
    
    # Get word tokens for later analysis in scikit
    X_train_sents = TextAnalysis(train).sentenceTokens('selftext')
    X_test_sents = TextAnalysis(test).sentenceTokens('selftext')
    
    # Train word2vec model again to get Word Embedding Vectors
    model_train_200 = Word2Vec(X_train_sents, size=200, window=5, min_count=10, workers=4, sg = 1)
    model_test_200 = Word2Vec(X_test_sents, size=200, window=5, min_count=10, workers=4, sg = 1)
    model_train_400 = Word2Vec(X_train_sents, size=400, window=5, min_count=10, workers=4, sg = 1)
    model_test_400 = Word2Vec(X_test_sents, size=400, window=5, min_count=10, workers=4, sg = 1)
    
    # Average word vectors
    w2v_train_200 = dict(zip(model_train_200.wv.index2word, model_train_200.wv.syn0))
    w2v_test_200 = dict(zip(model_test_200.wv.index2word, model_test_200.wv.syn0))
    w2v_train_400 = dict(zip(model_train_400.wv.index2word, model_train_400.wv.syn0))
    w2v_test_400 = dict(zip(model_test_400.wv.index2word, model_test_400.wv.syn0))
    
    # Get word tokens for later scikit modeling.
    train, test = train_test_split(data, test_size = 0.2, random_state = 42)
    trans_tokens_train = TextAnalysis(train).wordTokens('selftext')
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
    
    w2v_rf_200 = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v_train_200)), 
                        ("Random Forest", RandomForestClassifier(n_estimators=500, criterion='gini'))])
        
    w2v_rf_400 = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v_train_400)), 
                        ("Random Forest", RandomForestClassifier(n_estimators=500, criterion='gini'))])
    
    logistic_200 = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v_train_200)), 
                        ("Logistic Reg", linear_model.LogisticRegression())])
    
    logistic_400 = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v_train_400)), 
                        ("Logistic Reg", linear_model.LogisticRegression())])
    
    svc_200 = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v_train_200)),
                   ("linear svc", svm.SVC(kernel="linear", probability = True))])
    
    svc_400 = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v_train_400)),
                   ("linear svc", svm.SVC(kernel="linear", probability = True))]) 
    
    gradient_200 = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v_train_200)),
                   ("gradient boost", GradientBoostingClassifier(n_estimators = 500, learning_rate=1.0))]) 
    
    gradient_400 = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v_train_400)),
                   ("gradient boost", GradientBoostingClassifier(n_estimators = 500, learning_rate=1.0))]) 
    

    rf_Bow_200 = Pipeline([('tfidf', TfidfVectorizer(analyzer=lambda x: x,max_features = 200)), ('to dense', DenseTransformer()),
                        ("Random Forest", RandomForestClassifier(n_estimators=500, criterion='gini'))])
    
    rf_Bow_400 = Pipeline([('tfidf', TfidfVectorizer(analyzer=lambda x: x,max_features = 400)), ('to dense', DenseTransformer()),
                        ("Random Forest", RandomForestClassifier(n_estimators=500, criterion='gini'))])
                
    logistic_BoW_200 = Pipeline([('tfidf', TfidfVectorizer(analyzer=lambda x: x,max_features = 200)), ('to dense', DenseTransformer()), 
                        ("Logistic Reg", linear_model.LogisticRegression())])
    
    logistic_BoW_400 = Pipeline([('tfidf', TfidfVectorizer(analyzer=lambda x: x,max_features = 400)), ('to dense', DenseTransformer()), 
                        ("Logistic Reg", linear_model.LogisticRegression())])
    
    svc_BoW_200 = Pipeline([('tfidf', TfidfVectorizer(analyzer=lambda x: x,max_features = 200)), ('to dense', DenseTransformer()),
                   ("linear svc", svm.SVC(kernel="linear", probability = True))]) 
    
    svc_BoW_400 = Pipeline([('tfidf', TfidfVectorizer(analyzer=lambda x: x,max_features = 400)), ('to dense', DenseTransformer()),
                   ("linear svc", svm.SVC(kernel="linear", probability = True))]) 
    
    gradient_BoW_200 = Pipeline([('tfidf', TfidfVectorizer(analyzer=lambda x: x,max_features = 200)), ('to dense', DenseTransformer()),
                   ("gradient boost", GradientBoostingClassifier(n_estimators = 500, learning_rate=1.0))])
    
    gradient_BoW_400 = Pipeline([('tfidf', TfidfVectorizer(analyzer=lambda x: x,max_features = 400)), ('to dense', DenseTransformer()),
                   ("gradient boost", GradientBoostingClassifier(n_estimators = 500, learning_rate=1.0))]) 
    

    all_models = [("Random Forest W2V (200)", w2v_rf_200),("Logistic Regression W2V (200)", logistic_200), ("Linear SVC W2V (200)", svc_200), ("Gradient Boosted Trees W2V (200)", gradient_200), ("Random Forest W2V (400)", w2v_rf_400),("Logistic Regression W2V (400)", logistic_400), ("Linear SVC W2V (400)", svc_400), ("Gradient Boosted Trees W2V (400)", gradient_400), ("Random Forest BoW (200)", rf_Bow_200),("Logistic Reg BoW (200)", logistic_BoW_200), ("Linear SVC BoW (200)", svc_BoW_200), ("Gradient Boosted Trees BoW (200)", gradient_BoW_200), ("Random Forest BoW (400)", rf_Bow_400),("Logistic Reg BoW (400)", logistic_BoW_400), ("Linear SVC BoW (400)", svc_BoW_400), ("Gradient Boosted Trees BoW (400)", gradient_BoW_400)]
    
    scores = sorted([(name, cross_val_score(model, trans_tokens_train, y_train, cv=5).mean()) 
                     for name, model in all_models], 
                    key=lambda (_, x): -x)
    
    
    print "\n" + tabulate(scores, floatfmt=".4f", headers=("model", 'score'))

    
    plt.figure(figsize=(24, 8), dpi = 150)
    sns.set(font_scale=1.5)
    ax = sns.barplot(x=[name for name, _ in scores], y=[score for _, score in scores])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title("Model Classification Accuracy (KFold = 5)", fontsize = 24)
    ax.set_xlabel('Classification Model Name', fontsize = 1)
    ax.set_ylabel('Accuracy (Proportion Correct)', fontsize = 24)
    plt.ylim(.8,1)
    
    def rocData(model, x_train, x_test, y_train, y_test):
        model_fit = model.fit(x_train,y_train)
        pred = model_fit.predict_proba(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, np.transpose(pred)[1])
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc
    
    svc_bow_400_fpr, svc_bow_400_tpr, svc_bow_400_AUC = rocData(svc_BoW_400, trans_tokens_train, trans_tokens_test, y_train, y_test)
    rf_bow_400_fpr, rf_bow_400_tpr, rf_bow_400_AUC = rocData(rf_Bow_400, trans_tokens_train, trans_tokens_test, y_train, y_test)
    gradient_bow_fpr, gradient_bow_tpr, gradient_bow_AUC = rocData(gradient_BoW_400, trans_tokens_train, trans_tokens_test, y_train, y_test)
    rf_bow_200_fpr, rf_bow_200_tpr, rf_bow_200_AUC = rocData(rf_Bow_200, trans_tokens_train, trans_tokens_test, y_train, y_test)
    svc_bow_200_fpr, svc_bow_200_tpr, svc_bow_200_AUC = rocData(svc_BoW_200, trans_tokens_train, trans_tokens_test, y_train, y_test)
    gradient_bow_200_fpr, gradient_bow_200_tpr, gradient_bow_200_AUC = rocData(gradient_BoW_200, trans_tokens_train, trans_tokens_test, y_train, y_test)
    #SDG_bow_400_fpr, SDG_bow_400_tpr, SDG_bow_400_AUC = rocData(SDG_BoW_400, trans_tokens_train, trans_tokens_test, y_train, y_test)
    LR_bow_400_fpr, LR_bow_400_tpr, LR_bow_400_AUC = rocData(logistic_BoW_400, trans_tokens_train, trans_tokens_test, y_train, y_test)
    gradient_w2v_400_fpr, gradient_w2v_400_tpr, gradient_w2v_400_AUC = rocData(gradient_400, trans_tokens_train, trans_tokens_test, y_train, y_test)
    LR_bow_w2v_200_fpr, LR_bow_w2v_200_tpr, LR_bow_w2v_200_AUC = rocData(logistic_200, trans_tokens_train, trans_tokens_test, y_train, y_test)
    
    # Plot ROC curve
    plt.figure(dpi = 250)
    plt.plot(svc_bow_200_fpr, svc_bow_200_tpr, label='SVC BoW (200) (AUC = %0.3f)' % svc_bow_200_AUC)
    plt.plot(svc_bow_400_fpr, svc_bow_400_tpr, label='SVC BoW (400) (AUC = %0.3f)' % svc_bow_400_AUC)
    plt.plot(rf_bow_400_fpr, rf_bow_400_tpr, label='RF BoW (400) (AUC = %0.3f)' % rf_bow_400_AUC)
    plt.plot(gradient_bow_fpr, gradient_bow_tpr, label='Gradient Boost BoW (400) (AUC = %0.3f)' % gradient_bow_AUC)
    plt.plot(rf_bow_200_fpr, rf_bow_200_tpr, label='RF BoW (200) (AUC = %0.3f)' % rf_bow_200_AUC)
    plt.plot(LR_bow_400_fpr, LR_bow_400_tpr, label='Logistic BoW (400) (AUC = %0.3f)' % LR_bow_400_AUC)
    plt.plot(gradient_w2v_400_fpr, gradient_w2v_400_tpr, label='Gradient Boost W2V (400) (AUC = %0.3f)' % gradient_w2v_400_AUC)
    plt.plot(LR_bow_w2v_200_fpr, LR_bow_w2v_200_tpr, label='Logistic W2V (400) (AUC = %0.3f)' % LR_bow_w2v_200_AUC)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right", fontsize = 10)    
