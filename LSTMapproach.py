#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:58:47 2019

@author: sneha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

import RNN

df = pd.read_csv("/home/sneha/Documents/API_for_AntiCyberBullying-master/train.csv")
dft=pd.read_csv("/home/sneha/Documents/API_for_AntiCyberBullying-master/test_with_solutions.csv")
df.head()
dft.head()
df.drop(['Date'],axis=1,inplace=True)
dft.drop(['Date','Usage'],axis=1,inplace=True)

sns.countplot(df.Insult)
plt.xlabel('Label')
plt.title('Number of non-bully vs bully messages in trianing dataset')

X = df.Comment
Y = df.Insult
Y
le = LabelEncoder()
Y = le.fit_transform(Y)
Y
Y = Y.reshape(-1,1)
Y

Xt = df.Comment
Yt = df.Insult
Yt
le = LabelEncoder()
Yt = le.fit_transform(Y)
Yt
Yt = Yt.reshape(-1,1)
Yt

#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)
X_train,X_test,Y_train,Y_test=X,Xt,Y,Yt



max_words = 1000
max_len = 100
tok = Tokenizer(num_words=max_words,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)

sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
sequences_matrix

model = RNN.RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

X_test=querystring=["absolutely "]
#,['you are fucking stupid'],['you are a fat ass bully'],['you need to die because you are a useless who deserves to die'],['you need to dies because you deserves to die'],['you are mean']]
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

ans=model.predict(test_sequences_matrix,batch_size=None,verbose=0,steps=None)
print(ans)

#accr = model.evaluate(test_sequences_matrix,Y_test)

#print(accr)
#print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
