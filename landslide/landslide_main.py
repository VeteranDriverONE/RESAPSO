# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 23:01:41 2021

@author: Administrator
"""

import os
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def landslide(para): 
    # read training data 
    df = pd.read_excel('data.xlsx',header=None)
    ''' 
    'data.xlsx' is dataset of landslide, which has N row and 11 column,
    N denotes number of sample, 11 column including 10 feature and one label in our paper.
    Each row feature is the offset of the current month value and the previous month value.
    the details of the 10 features are in our paper.
    label（displacement distance） denotes the Euclidean distance between 
    the sensor position of the current month and the previous month.
    '''
    # bulid model from 'para'     
    form=[5,5]
    var_bin=list()
    for i in range(len(form)):
        re=int(para[i].round())
        #print(re)
        temp_var_bin=list()
        for i in range(form[i]):
            temp_var_bin.append(re & 1)
            re=re >> 1
        temp_var_bin.reverse()
        var_bin=var_bin+temp_var_bin
    for i in range(len(var_bin)):
        if var_bin[i]==0:
            del df[i]
    for col in df.iteritems():
        #print(col[1])
        #type(col[1])
        max=col[1].max()
        min=col[1].min()
        df[col[0]]=(col[1]-min)/(max-min)
    dataset = df
    dataset = dataset.values

    n_steps = int(para[2].round())
    X, y = split_sequences(dataset, n_steps)
    #print(X.shape, y.shape)
    bound=round(X.shape[0]*0.7) # Ratio of training and testing sets
    x_train=X[:bound]
    y_train=y[:bound]
    x_test=X[bound:]
    y_test=y[bound:]

    if para[4].round()==0:
        act='sigmoid'
    elif para[4].round()==1:
        act='tanh'
    elif para[4].round()==2:
        act='relu'

    n_features = x_train.shape[2]
    #print(n_features)
    model = Sequential()

    model.add(LSTM(int(para[3].round()), activation = act,return_sequences = False, input_shape = (n_steps,n_features)))

    #model.add(LSTM(16, activation = 'tanh',return_sequences = False))
    #model.add(Dropout(0.3))
    #model.add(Dense(32,activation = 'tanh'))
    #model.add(Dense(32,activation = 'tanh'))
    #model.add(Dropout(0.3))

    model.add(Dense(1,activation = 'tanh'))
    model.compile(loss = 'mse', optimizer = 'adam')
    
    #print(x_train)
    #print(y_train)
    print(para)
    history = model.fit(x_train, y_train, epochs = 128, batch_size = 5, verbose = 1, shuffle = False)

    y_pred = model.predict(x_test)

    real=dataset[:,-1]*(max-min)+min
    pred=y_pred*(max-min)+min
    pred2=y_pred*(max-min)+min
    for i in range(len(dataset)):
        if i==0:
            real[i]=real[i]+8.2006
        else:
            real[i]=real[i]+real[i-1]
        if i>=bound+n_steps-1:
            pred[i-bound-n_steps+1]=pred[i-bound-n_steps+1]+real[i-1]
        if i==bound+n_steps-1:
            pred2[i-bound-n_steps+1]=pred2[i-bound-n_steps+1]+real[i-1]
        elif i>=bound+n_steps-1:
            pred2[i-bound-n_steps+1]=pred2[i-bound-n_steps+1]+pred2[i-bound-n_steps]
    
    #plotFigure(real,y_pred,y_test,pred,pred2)
    
    ls=list()
    ls.append(real) # 真实的所有月的移值
    ls.append(y_pred) # 预测的位移值
    ls.append(np.expand_dims(y_test, axis=1)) # 真实的测试月的位移值
    ls.append(pred) # 预测值，上一个月的真实值+预测的位移值
    ls.append(pred2) # 预测值，上一个月的预测值+预测的位移值
    #for x in locals().keys():
        #print(locals()[x])
    return ls

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
    # find the end of this pattern
        end_ix = i + n_steps
    # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
    # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def plotFigure(real,y_pred,y_test,pred,pred2,bound,n_steps):
    plt.plot(y_test,c='black', label = "real value")
    plt.plot(y_pred,color='black', label = "predicted value",linestyle="--") 
    plt.xlabel('test sample')
    plt.ylabel('displacement distance')
    plt.legend(['real value','predicted value'])
    plt.show()
    
    bound=int(bound)
    print(bound)
    xtick=[0,10,20,30,40];
    xlabel=list()
    for i in range(len(xtick)):
        ye=math.floor((xtick[i]+bound+3+n_steps-1)/12)+2007
        mo=(xtick[i]+bound+3+n_steps-1)%12
        xlabel.append(str(ye)+'-'+str(int(mo)))

    plt.plot(real[-len(pred2):],c='black', label = "real value")
    plt.plot(pred,color='red', label = "predicted value",linestyle="--") 
    plt.suptitle('Fitting curve based on real value',fontsize=16)
    ax=plt.gca()
    ax.set_xticks(xtick)
    ax.set_xticklabels(xlabel, rotation=30, fontsize=12)
    plt.xlabel('Date',fontsize=12)
    plt.ylabel('displacement distance',fontsize=12)
    plt.legend(['real value','predicted value'])
    plt.show()

    plt.plot(real[-len(pred2):],c='black', label = "real value")
    plt.plot(pred2,color='red', label = "predicted value",linestyle="--") 
    plt.suptitle('Fitting curve based on predicted value',fontsize=16)
    ax=plt.gca()
    ax.set_xticks(xtick)
    ax.set_xticklabels(xlabel, rotation=30, fontsize=12)
    plt.xlabel('Date',fontsize=12)
    plt.ylabel('displacement distance',fontsize=12)
    plt.legend(['real value','predicted value'])
    plt.show()

def garbage_collect():
    for key in list(locals().keys()):
        if not key.startswith('__'):
            del locals()[key]
