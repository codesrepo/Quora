# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 18:34:55 2017

@author: Administrator
"""

import json
from pprint import pprint
import pandas as pd
from random import randint

with open('C:\\Users\\Administrator\\Downloads\\train-v1.1.json') as data_file:    
    data = json.load(data_file)

#pprint(data)

data["data"][440]["paragraphs"][0]['qas'][2]['question']

title = []
paragraph = []
question = []
for pos1 in range(0,440):
    current_title = data["data"][pos1]["title"]
    para = 0
    for pos2 in range(0,100):
        current_para = para + 1   
        for pos3 in range(0,100):
            try:
                current_question = data["data"][pos1]["paragraphs"][pos2]['qas'][pos3]['question']
                question.append(current_question)
                paragraph.append(current_title+"_"+str(pos2))
                title.append(current_title)
            except:
                continue
            
unique_paragraphs = list(set(paragraph))

df_raq = pd.DataFrame({"Title":title,"paragraph":paragraph,"question":question})             


def createQuestions(qlist):
    q1=  []
    q2 = []
    for i in qlist:
        for j in qlist:
            if str(i)!=str(j):
                #print (i,j)
                q1.append(str(i))
                q2.append(str(j))
    return q1,q2
    
count = 0 
q1 = []
q2 = []           
while count <448750:  
    rand_index = randint(0,18676)
    selected_para = unique_paragraphs[rand_index]
    temp = df_raq[df_raq.paragraph==selected_para]
    a,b = createQuestions(temp.question.tolist())
    q1.extend(a)
    q2.extend(b)      
    count = count +len(a)
              
print("hi")    
      
train_extra = pd.DataFrame({"question1":q1,"question2":q2})            
train_extra['is_duplicate'] = 0
train_extra['id'] = 0
train_extra['qid1'] = 0
train_extra['qid2'] = 0
train_extra.to_csv("C:/Quora/train_extra.csv",index =False)
df = pd.read_csv(TRAIN_DATA_FILE)        
df_new = pd.concat([df,train_extra],axis = 0)

df_new.to_csv("C:/Quora/train_additional_v04232017.csv",index =False)
