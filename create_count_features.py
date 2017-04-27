import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.2.0-posix-seh-rt_v5-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import pandas as pd
import numpy as np
import nltk
from collections import Counter
#from nltk.corpus import stopwords

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.optimize import minimize
stops = set(["a","an","the"])
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import multiprocessing
import difflib
import re
import codecs
from nltk.stem import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity 
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors
from collections import Counter
from sklearn.decomposition import PCA
#from nltk.tag.perceptron import PerceptronTagger

import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#tagger = PerceptronTagger()
BASE_DIR = 'C:/Quora/'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
A_CONSTANT = 0.0003  

print('Indexing word vectors')
stops = set(stopwords.words("english"))
        
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

avg = np.array([0]*300).astype(np.float32) 
  
def parse_sent_word2vec(row):
    try:
        q1_text = str(row['question1']).lower().split()
        q2_text = str(row['question2']).lower().split()
        a1=[]
        a2=[]
        for  word in   q1_text:
            temp = avg
            P_W = TOKEN_COUNT.get(word,0)/float(TOTAL_TOKEN)
            try:
                temp = word2vec.word_vec(word)*(A_CONSTANT/(A_CONSTANT+P_W))
            except:
                continue
                
            a1.append(temp)
        for  word in   q2_text:
            temp = avg
            P_W = TOKEN_COUNT.get(word,0)/float(TOTAL_TOKEN)
            try:
                temp = word2vec.word_vec(word)*(A_CONSTANT/(A_CONSTANT+P_W))
            except:
                continue
            a2.append(temp)
                
            
        
        parsed_vector1= np.mean(np.array(a1),axis = 0)
        parsed_vector2= np.mean(np.array(a2),axis = 0) 
        if np.sum(parsed_vector1)==0 or np.sum(parsed_vector2)==0:
            return -1
         
        parsed_vector1 = parsed_vector1 - pca.transform(parsed_vector1)
       
        parsed_vector2 = parsed_vector2 - pca.transform(parsed_vector2)
    
        #print (parsed_vector1)
        #print (parsed_vector2)
    except:
        return -1
    
    #parsed_vector1= np.sum(np.array([word2vec.word_vec(word) for word in   q1_text]))   
    #parsed_vector2= np.sum(np.array([word2vec.word_vec(word) for word in   q2_text])) 
    
    #print (parsed_vector1)
    #print (parsed_vector2)
    try:
        cs = cosine(parsed_vector1,parsed_vector2)
        #print (cs)
    except:
        return -2
    
    return cs
    
def getSentenceVector(row):
    try:
        q1_text = str(row).lower().split()   
        a1=[]    
        for  word in   q1_text:
            temp = avg
            P_W = TOKEN_COUNT.get(word,0)/float(TOTAL_TOKEN)
            try:
                temp = word2vec.word_vec(word)*(A_CONSTANT/(A_CONSTANT+P_W))
            except:
                continue
            a1.append(temp)    
        
        parsed_vector1= np.mean(np.array(a1),axis = 0)
    except:
            return avg
    return parsed_vector1

    
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\0k ", "0000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

texts_1 = [] 
texts_2 = []
labels = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(text_to_wordlist(values[3]))
        texts_2.append(text_to_wordlist(values[4]))
        if int(values[0])%1000==0:
            print ("Already processed %d train files!!!"%(int(values[0])))
        labels.append(int(values[1]))
print('Found %s texts in train.csv' % len(texts_1))

test_texts_1 = []
test_texts_2 = []
test_ids = []
with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(text_to_wordlist(values[1]))
        test_texts_2.append(text_to_wordlist(values[2]))
        if int(values[0])%1000==0:
            print ("Already processed %d test files!!!"%(int(values[0])))
        test_ids.append(values[0])
print('Found %s texts in test.csv' % len(test_texts_1))

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)
train["question1"]=texts_1
train["question2"]=texts_2

    
    

test["question1"]=test_texts_1
test["question2"]=test_texts_2

tfidf_txt = pd.Series(texts_1 + texts_2 + test_texts_1 + test_texts_2).astype(str)
qcount  =  Counter(tfidf_txt.tolist())

train["question1c"]=train.question1.map(lambda x: qcount[x])
train["question2c"]=train.question2.map(lambda x: qcount[x])

test["question1c"]=test.question1.map(lambda x: qcount[x])
test["question2c"]=test.question2.map(lambda x: qcount[x])

np.save("C:\\Quora\\train_f1.npy",train.question1c.tolist())
np.save("C:\\Quora\\train_f2.npy",train.question2c.tolist())

np.save("C:\\Quora\\test_f1.npy",test.question1c.tolist())
np.save("C:\\Quora\\test_f2.npy",test.question2c.tolist())
