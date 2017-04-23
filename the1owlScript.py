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
TRAIN_DATA_FILE = BASE_DIR + 'train_additional_v04232017.csv'
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
        texts_1.append(text_to_wordlist(values[4]))
        texts_2.append(text_to_wordlist(values[5]))
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
#
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
#cvect = CountVectorizer(stop_words='english', ngram_range=(1, 1))



tfidf_txt = pd.Series(texts_1 + texts_2 + test_texts_1 + test_texts_2).astype(str)
tr_qs = pd.Series(train["question1"] + train["question2"])
tfidf.fit_transform(tfidf_txt)
temp = tfidf.transform(tfidf_txt)
#cvect.fit_transform(tfidf_txt)
tokens = " ".join(tfidf_txt.tolist()).split(" ")
TOKEN_COUNT =Counter(tokens)
TOTAL_TOKEN = len(tokens)
del tokens

arr_train_q1 = np.zeros(shape=(len(texts_1),300))
c = 0
for s in texts_1:
    arr_train_q1[c] = getSentenceVector(s)
    c = c+1
arr_train_q2 = np.zeros(shape=(len(texts_1),300))
c = 0
for s in texts_2:
    arr_train_q2[c] = getSentenceVector(s)
    c = c+1


print("Creating test word vectors")
arr_test_q1 = np.zeros(shape=(len(test_texts_1),300))
c = 0
for s in test_texts_1:
    arr_test_q1[c] = getSentenceVector(s)
    c = c+1
arr_test_q2 = np.zeros(shape=(len(test_texts_2),300))
c = 0
for s in test_texts_2:
    arr_test_q2[c] = getSentenceVector(s)
    c = c+1
# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
weights = {word: get_weight(count) for word, count in TOKEN_COUNT.items()}

def sen2Vec(s):
    try:
        q1_text = str(s).lower().split()
        
        a1=[]
        
        for  word in   q1_text:
            temp = avg
            P_W = TOKEN_COUNT.get(word,0)/float(TOTAL_TOKEN)
            #print(word,P_W)
            try:
                temp = word2vec.word_vec(word)*(A_CONSTANT/(A_CONSTANT+P_W))
            except:
                continue
            a1.append(temp)         
        
        parsed_vector1= np.mean(np.array(a1),axis = 0)   
        #print(parsed_vector1)
    except:
        return None
    return parsed_vector1
del test_texts_1,test_texts_2,texts_1,texts_2  
import gc
gc.collect()

sentence_vectors = []
X = np.zeros(shape=(500000,300))

count = 0
for s in tfidf_txt.tolist():
    v = sen2Vec(s)
    #print(v)
    if v!=None and count <500000:
        #sentence_vectors.append(v)
        X[count] = v
        count = count+1


pca = PCA(n_components=1)
X=X[~np.isnan(X).any(axis=1)]
pca.fit(X) 
print(pca.explained_variance_ratio_) 

arr_train_q1[np.isnan(arr_train_q1)]=0
q1_delta = pca.transform(arr_train_q1)
arr_train_q1 = arr_train_q1-q1_delta*arr_train_q1

arr_train_q2[np.isnan(arr_train_q2)]=0
q2_delta = pca.transform(arr_train_q2)
arr_train_q2 = arr_train_q2-q2_delta*arr_train_q2

train_angle = []
for i in range(0,len(train)):
    temp=cosine(arr_train_q1[i],arr_train_q2[i])
    if temp <0.1 or temp >0.8:
        #print temp
    train_angle.append(temp)
    
arr_test_q1[np.isnan(arr_test_q1)]=0
q1_delta = pca.transform(arr_test_q1)
arr_test_q1 = arr_test_q1-q1_delta*arr_test_q1

arr_test_q2[np.isnan(arr_test_q2)]=0
q2_delta = pca.transform(arr_test_q2)
arr_test_q2 = arr_test_q2-q2_delta*arr_test_q2

test_angle = []
for i in range(0,len(test)):
    temp=cosine(arr_test_q1[i],arr_test_q2[i])
    if temp <0.1 or temp >0.9:
        #print temp
    test_angle.append(temp)    

train['cosine_similarity'] = train_angle
test['cosine_similarity'] = test_angle
del tfidf_txt,sentence_vectors,arr_train_q1,arr_train_q2,train_angle,arr_test_q2,arr_test_q1,test_angle,pca




a = 0.165 / 0.37
b = (1 - 0.165) / (1 - 0.37)
def kappa(preds, y):
    score = []
    for pp,yy in zip(preds, y.get_label()):
        score.append(a * yy * np.log (pp) + b * (1 - yy) * np.log(1-pp))
    score = -np.sum(score) / len(score)

    return 'kappa', float(score)

def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(st1).lower(), str(st2).lower())
    return seq.ratio()

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R
    
def word_match_begin_end(row):
    R=-1
    try:
    
        q1word_start = str(row['question1']).lower().split()[0]
        q2word_start = str(row['question2']).lower().split()[0]
    
        q1word_end = str(row['question1']).lower().split()[-1]
        q2word_end = str(row['question2']).lower().split()[-1]
    except:
        return 1000
    
    #print("q1 strat,q2 strat,q1 end,q2 end",q1word_start,q2word_start,q1word_end,q2word_end)
    
    if q1word_start == q2word_start and q1word_end == q2word_end and q1word_start=="what":
        R= 1
    
    elif q1word_start == q2word_start and q1word_end == q2word_end:
        R= 2
        
    elif q1word_start == q2word_start and q1word_start=="what":
        R= 3
        
    elif q1word_start == q2word_start:
        R= 4
        
    elif q1word_end == q2word_end:
        R= 5 
    #print("R=",R)
    return R
def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R
def first_word(row):
    try:
        q1_text = str(row['question1']).lower().split()[0]
        q2_text = str(row['question2']).lower().split()[0]
        return q1_text+"_"+q2_text
    except:
        return "None_None"

le = preprocessing.LabelEncoder()
def get_features(df_features):
#    print('nouns...')
#    df_features['question1_nouns'] = df_features.question1.map(lambda x: [w for w, t in tagger.tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
#    df_features['question2_nouns'] = df_features.question2.map(lambda x: [w for w, t in tagger.tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
#    df_features['z_noun_match'] = df_features.apply(lambda r: sum([1 for w in r.question1_nouns if w in r.question2_nouns]), axis=1)  #takes long
    print('lengths...')
    
    df_features['z_len1'] = df_features.question1.map(lambda x: len(str(x)))
    df_features['z_len2'] = df_features.question2.map(lambda x: len(str(x)))
    df_features['z_word_len1'] = df_features.question1.map(lambda x: len(str(x).split()))
    df_features['z_word_len2'] = df_features.question2.map(lambda x: len(str(x).split()))
    print('india/indian')
    df_features['question1_india'] = df_features.question1.map(lambda x: 'india' in str(x))
    df_features['question2_india'] = df_features.question2.map(lambda x: 'india' in str(x))
    df_features['indiacator'] = df_features['question1_india'] + df_features['question2_india']
    print('difflib...')
    df_features['z_match_ratio'] = df_features.apply(lambda r: diff_ratios(r.question1, r.question2), axis=1)  #takes long
    print('word match...')
    df_features['z_word_match'] = df_features.apply(word_match_share, axis=1, raw=True)
    df_features['z_magic_feature'] = df_features.apply(word_match_begin_end, axis=1, raw=True)
    df_features['z_magic_feature_'] = df_features.apply(tfidf_word_match_share, axis=1, raw=True)
    
    z1 = tfidf.transform(df_features.question1)
    z2 = tfidf.transform(df_features.question2)
    print('tfidf...z_tfidf_sum1')

    df_features['z_tfidf_sum1'] = z1.sum(axis = 1)
    print('tfidf...z_tfidf_sum2')
    df_features['z_tfidf_sum2'] = z2.sum(axis = 1)
    df_features['z_tfidf_len1'] = z1.sign().sum(axis = 1)
    df_features['z_tfidf_len2'] = z2.sign().sum(axis = 1)
    df_features['z_tfidf_mean1'] = df_features['z_tfidf_sum1']/df_features['z_tfidf_len1']
    df_features['z_tfidf_mean2'] = df_features['z_tfidf_sum2']/df_features['z_tfidf_len2']
    
    
    
    
#    print('tfidf...z_tfidf_sum1')
#    df_features['z_tfidf_sum1'] = df_features.question1.map(lambda x: np.sum(tfidf.transform([str(x)]).data))
#    print('tfidf...z_tfidf_sum2')
#    df_features['z_tfidf_sum2'] = df_features.question2.map(lambda x: np.sum(tfidf.transform([str(x)]).data))
#    print('tfidf...z_tfidf_mean1')
#    df_features['z_tfidf_mean1'] = df_features.question1.map(lambda x: np.mean(tfidf.transform([str(x)]).data))
#    print('tfidf...z_tfidf_mean2')
#    df_features['z_tfidf_mean2'] = df_features.question2.map(lambda x: np.mean(tfidf.transform([str(x)]).data))
#    print('tfidf...z_tfidf_len1')
#    df_features['z_tfidf_len1'] = df_features.question1.map(lambda x: len(tfidf.transform([str(x)]).data))
#    print('tfidf...z_tfidf_len2')
#    df_features['z_tfidf_len2'] = df_features.question2.map(lambda x: len(tfidf.transform([str(x)]).data))
    return df_features.fillna(0.0)

train['magic_feature'] = train.apply(word_match_begin_end, axis=1, raw=True)
train = get_features(train)
train_features = pd.read_csv("C:\\Quora\\train_features.csv")
train_features.drop(['question1','question2'],axis=1,inplace=True)
train = pd.concat([train,train_features], axis=1)
print ("Super awesome feature")
train['super_awesome_feature'] = train.apply(first_word, axis=1, raw=True)
test_saf = test.apply(first_word, axis=1, raw=True)
le.fit(train['super_awesome_feature'].tolist()+list(test_saf))
train['super_awesome_feature'] = le.transform(train['super_awesome_feature'])
train.to_csv('train_XGB.csv', index=False)

#pd.to_pickle(train['super_awesome_feature'],"C:/Quora/train_super.pkl")
#pd.to_pickle(train['z_magic_feature'] ,"C:/Quora/teain_magic1.pkl")

col = [c for c in train.columns if c not in ['is_duplicate','question1','question2','id','qid1','qid2']]
#
#pos_train = train[train['is_duplicate'] == 1]
#neg_train = train[train['is_duplicate'] == 0]
#p = 0.175
#scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
#while scale > 1:
#    neg_train = pd.concat([neg_train, neg_train])
#    scale -=1
#neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
#train = pd.concat([pos_train, neg_train])

x_train, x_valid, y_train, y_valid = train_test_split(train[col], train['is_duplicate'], test_size=0.2, random_state=0)

#x_train = train[train.id>=80000]
#y_train = x_train['is_duplicate']
#x_train=x_train[col]
#x_valid = train[train.id<80000]
#y_valid = x_valid['is_duplicate']
#x_valid=x_valid[col]








params = {}
params["objective"] = "binary:logistic"
params['eval_metric'] = 'logloss'
params["eta"] = 0.1
params["subsample"] = 0.9
params["min_child_weight"] = 1
params["colsample_bytree"] = 0.7
params["max_depth"] = 6
params["silent"] = 1
params["seed"] = 1632

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
bst = xgb.train(params, d_train, 700000, watchlist, early_stopping_rounds=500, verbose_eval=100) #change to higher #s
print(log_loss(train.is_duplicate, bst.predict(xgb.DMatrix(train[col]))))

#test = get_features(test)
test['magic_feature'] = test.apply(word_match_begin_end, axis=1, raw=True)
test = get_features(test)
test_features = pd.read_csv("C:\\Quora\\test_features.csv")
test_features.drop(['question1','question2'],axis=1,inplace=True)
test = pd.concat([test,test_features], axis=1)
test['super_awesome_feature'] = test.apply(first_word, axis=1, raw=True)
test['super_awesome_feature'] = le.transform(test['super_awesome_feature'])
test.to_csv('test_XGB.csv', index=False)
#pd.to_pickle(test['super_awesome_feature'] ,"C:/Quora/test_super.pkl")
#pd.to_pickle(test['z_magic_feature'] ,"C:/Quora/test_magic1.pkl")


sub = pd.DataFrame()
sub['test_id'] = test['test_id']
sub['is_duplicate'] = bst.predict(xgb.DMatrix(test[col]))

sub.to_csv('z08_submission_xgb_03.csv', index=False)
