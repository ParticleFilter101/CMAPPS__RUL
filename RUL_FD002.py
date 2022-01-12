#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 20:32:45 2022

@author: shivam.bhardwaj
"""

import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression,LinearRegression,SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score,classification_report, confusion_matrix, mean_absolute_error, mean_squared_error,r2_score, silhouette_samples,silhouette_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,AdaBoostRegressor, BaggingRegressor
from sklearn.svm import SVR,LinearSVR,NuSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.layers.core import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Bidirectional, GRU, LSTM, RepeatVector, TimeDistributed,Conv1D,MaxPooling1D,ReLU,UpSampling1D,Input,LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from keras.layers.core import Dense
from keras.layers.merge import concatenate
from tensorflow.keras.layers import Average
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
import pywt
from scipy.fft import fft,ifft
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
# from kneed import KneeLocator
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('train_FD002.txt', sep = ' ', header = None)
data_act = data[[f for f in range(0, 26)]]
data_act.columns = ["ID", "Cycle", "OpSet1", "OpSet2", "OpSet3", "SensorMeasure1", "SensorMeasure2", "SensorMeasure3", "SensorMeasure4",
                "SensorMeasure5", "SensorMeasure6", "SensorMeasure7", "SensorMeasure8", "SensorMeasure9", "SensorMeasure10", "SensorMeasure11",
                "SensorMeasure12", "SensorMeasure13", "SensorMeasure14", "SensorMeasure15", "SensorMeasure16",
                "SensorMeasure17", "SensorMeasure18", "SensorMeasure19", "SensorMeasure20", "SensorMeasure21"]

data_op_set = data_act[['OpSet1','OpSet2','OpSet3']]
print(type(data_op_set))
op_set_ar = data_op_set.values
print(op_set_ar.shape)
scaler1 = StandardScaler()
op_set_ar_sc = scaler1.fit_transform(op_set_ar)

kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
   }

silhouette_coefficients = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(op_set_ar_sc)
    score = silhouette_score(op_set_ar_sc, kmeans.labels_)
    silhouette_coefficients.append(score)
    
kmeans = KMeans(n_clusters = 6, **kmeans_kwargs)
kmeans.fit(op_set_ar_sc)
label_tr = kmeans.predict(op_set_ar_sc)
print(label_tr.shape)
data_act['op_set_net'] = label_tr

max_cycles_df = data_act.groupby(["ID"], sort=False)["Cycle"].max().reset_index().rename(columns={"Cycle" : "MaxCycleID"})
FD002_df = pd.merge(data_act, max_cycles_df, how="inner", on="ID")
FD002_df["RUL"] = FD002_df["MaxCycleID"] - FD002_df["Cycle"]
grouped_k_m = FD002_df.groupby(['op_set_net']).mean()
print(grouped_k_m)
print(grouped_k_m.loc[0,'SensorMeasure1'])
grouped_k_std = FD002_df.groupby(['op_set_net']).std()
print(grouped_k_std)
print(grouped_k_std.loc[0,'SensorMeasure1'])

sensors = ['SensorMeasure2','SensorMeasure3', 'SensorMeasure4', 
           'SensorMeasure6','SensorMeasure7', 'SensorMeasure8',
           'SensorMeasure9', 'SensorMeasure11', 'SensorMeasure12',
           'SensorMeasure13', 'SensorMeasure14', 'SensorMeasure15',
           'SensorMeasure17','SensorMeasure20', 'SensorMeasure21']

def get_norm_df(main_df = FD002_df,mean_df = grouped_k_m, std_df = grouped_k_std, columns = sensors, num_index = FD002_df.shape[0]):
    norm_df = pd.DataFrame(index = range(num_index), columns = columns)
    for i in range(norm_df.shape[0]):
        for col in columns:
            if main_df.loc[i,'op_set_net'] == 0:
                norm_df.loc[i,col] = (main_df.loc[i,col] - mean_df.loc[0,col])/(std_df.loc[0,col])  
            elif main_df.loc[i,'op_set_net'] == 1:
                norm_df.loc[i,col] = (main_df.loc[i,col] - mean_df.loc[1,col])/(std_df.loc[1,col]) 
            elif main_df.loc[i,'op_set_net'] == 2:
                norm_df.loc[i,col] = (main_df.loc[i,col] - mean_df.loc[2,col])/(std_df.loc[2,col])
            elif main_df.loc[i,'op_set_net'] == 3:
                norm_df.loc[i,col] = (main_df.loc[i,col] - mean_df.loc[3,col])/(std_df.loc[3,col])
            elif main_df.loc[i,'op_set_net'] == 4:
                norm_df.loc[i,col] = (main_df.loc[i,col] - mean_df.loc[4,col])/(std_df.loc[4,col])
            else:
                norm_df.loc[i,col] = (main_df.loc[i,col] - mean_df.loc[5,col])/(std_df.loc[5,col]) 
    return norm_df


FD002_norm_df = get_norm_df()
FD002_norm_df[['ID','Cycle','op_set_net','RUL']] = FD002_df.loc[:,['ID','Cycle','op_set_net','RUL']]
enc_opset = OneHotEncoder()
opset_enc = enc_opset.fit_transform(FD002_norm_df['op_set_net'].values.reshape(-1,1)).toarray()
print(opset_enc.shape)

x_num_ar = FD002_norm_df[sensors].values
ID_ar = FD002_norm_df['ID'].values.reshape(-1,1)
y_ar = FD002_norm_df['RUL'].values.reshape(-1,1)
final_tr_df = pd.DataFrame(np.concatenate([ID_ar,opset_enc,x_num_ar,y_ar],axis = 1), 
                           columns = ['ID','enc_set1', 'enc_set2','enc_set3','enc_set4',
                                      'enc_set5','enc_set6','SensorMeasure2','SensorMeasure3',
                                      'SensorMeasure4', 'SensorMeasure6','SensorMeasure7', 
                                      'SensorMeasure8','SensorMeasure9', 'SensorMeasure11', 
                                      'SensorMeasure12','SensorMeasure13', 'SensorMeasure14',
                                      'SensorMeasure15','SensorMeasure17','SensorMeasure20', 
                                      'SensorMeasure21','RUL'])


def get_seq(in_arr,out_arr,lb):
    x,y = [],[]
    for i in range(in_arr.shape[0]):
        if (i+lb) <= in_arr.shape[0]:
            x.append(in_arr[i:(i+lb),:])
            y.append(out_arr[(i+lb)-1])
    x = np.array(x)
    y = np.array(y)
    return x,y

df_gb = final_tr_df.groupby(['ID'])
group =[]
for g, df_g in df_gb:
    print(g)
    group.append(g)
    print(len(df_g.axes[0]),len(df_g.axes[1]))
    print(df_g.columns)
print(len(group))  

in_ex,out_ex = [],[]
for g in group:
    df = df_gb.get_group(g)
    in_ar = df[['enc_set1', 'enc_set2','enc_set3','enc_set4',
                'enc_set5','enc_set6','SensorMeasure2','SensorMeasure3',
                'SensorMeasure4', 'SensorMeasure6','SensorMeasure7', 
                'SensorMeasure8','SensorMeasure9', 'SensorMeasure11', 
                'SensorMeasure12','SensorMeasure13', 'SensorMeasure14',
                'SensorMeasure15','SensorMeasure17','SensorMeasure20', 
                'SensorMeasure21']].values
    out_ar = df['RUL'].values
    x,y = get_seq(in_ar,out_ar,128)
    in_ex.append(x)
    out_ex.append(y)

x_tr = np.concatenate(in_ex)
print(x_tr.shape)
y_tr = np.concatenate(out_ex)
print(y_tr.shape)
x_tr = np.asarray(x_tr).astype(np.float32)
y_tr = np.asarray(y_tr).astype(np.float32)

test_data_FD002 = pd.read_csv('test_FD002.txt', sep = ' ', header = None)
test_data_FD002 = test_data_FD002[[f for f in range(0, 26)]]
test_data_FD002.columns = ["ID", "Cycle", "OpSet1", "OpSet2", "OpSet3", 
                           "SensorMeasure1", "SensorMeasure2", "SensorMeasure3", 
                           "SensorMeasure4","SensorMeasure5", "SensorMeasure6", 
                           "SensorMeasure7", "SensorMeasure8", "SensorMeasure9", 
                           "SensorMeasure10", "SensorMeasure11","SensorMeasure12", 
                           "SensorMeasure13", "SensorMeasure14", "SensorMeasure15", 
                           "SensorMeasure16","SensorMeasure17", "SensorMeasure18", 
                           "SensorMeasure19","SensorMeasure20", "SensorMeasure21"]
test_data_FD002.head()

max_cycles_df_test = test_data_FD002.groupby(["ID"], sort=False)["Cycle"].max().reset_index().rename(columns={"Cycle" : "MaxCycleID"})
y_test_rul_FD002 = pd.read_csv('RUL_FD002.txt', header = None)
y_test_rul_FD002.columns = ['Eng_rul']
y_test_rul_FD002['ID'] = max_cycles_df_test['ID']

max_cycles_df_test['MaxCycleIDAct'] = y_test_rul_FD002['Eng_rul']+ max_cycles_df_test['MaxCycleID']

test_df_FD002 = pd.merge(test_data_FD002, max_cycles_df_test, how="inner", on="ID")
test_df_FD002["RUL"] = test_df_FD002["MaxCycleIDAct"] - test_df_FD002["Cycle"]

data_op_set_te = test_df_FD002[['OpSet1','OpSet2','OpSet3']]
print(type(data_op_set_te))
op_set_ar_te = data_op_set_te.values
op_set_ar_sc_te = scaler1.transform(op_set_ar_te)

label_te = kmeans.predict(op_set_ar_sc_te)
print(label_te.shape)

test_df_FD002['op_set_net'] = label_te

test_df_FD002_norm = get_norm_df(main_df=test_df_FD002, num_index=test_df_FD002.shape[0])
test_df_FD002_norm[['ID','Cycle','op_set_net','RUL']] = test_df_FD002.loc[:,['ID','Cycle','op_set_net','RUL']]
opset_enc_te = enc_opset.transform(test_df_FD002_norm['op_set_net'].values.reshape(-1,1)).toarray()
print(opset_enc.shape)
x_num_ar_te = test_df_FD002_norm[sensors].values
ID_ar_te = test_df_FD002_norm['ID'].values.reshape(-1,1)
y_ar_te = test_df_FD002_norm['RUL'].values.reshape(-1,1)
final_te_df = pd.DataFrame(np.concatenate([ID_ar_te,opset_enc_te,x_num_ar_te,y_ar_te],axis = 1), 
                           columns = ['ID','enc_set1', 'enc_set2','enc_set3','enc_set4',
                                      'enc_set5','enc_set6','SensorMeasure2','SensorMeasure3',
                                      'SensorMeasure4', 'SensorMeasure6','SensorMeasure7', 
                                      'SensorMeasure8','SensorMeasure9', 'SensorMeasure11', 
                                      'SensorMeasure12','SensorMeasure13', 'SensorMeasure14',
                                      'SensorMeasure15','SensorMeasure17','SensorMeasure20', 
                                      'SensorMeasure21','RUL'])
df_gb_te = final_te_df.groupby(['ID'])
group_te =[]
for g, df_g in df_gb_te:
    print(g)
    group_te.append(g)
    print(len(df_g.axes[0]),len(df_g.axes[1]))
    print(df_g.columns)
print(len(group_te))  

in_ex_te,out_ex_te = [],[]
for g in group_te:
    df = df_gb_te.get_group(g)
    in_ar = df[['enc_set1', 'enc_set2','enc_set3','enc_set4',
                'enc_set5','enc_set6','SensorMeasure2','SensorMeasure3',
                'SensorMeasure4', 'SensorMeasure6','SensorMeasure7', 
                'SensorMeasure8','SensorMeasure9', 'SensorMeasure11', 
                'SensorMeasure12','SensorMeasure13', 'SensorMeasure14',
                'SensorMeasure15','SensorMeasure17','SensorMeasure20', 
                'SensorMeasure21']].values
    out_ar = df['RUL'].values
    x,y = get_seq(in_ar,out_ar,128)
    in_ex_te.append(x)
    out_ex_te.append(y)


in_ex_te_1, out_ex_te_1 = [], []
for i in range(len(in_ex_te)):
    if in_ex_te[i].ndim ==3:
        in_ex_te_1.append(in_ex_te[i])
        out_ex_te_1.append(out_ex_te[i])
        
x_te = np.concatenate(in_ex_te_1)
print(x_te.shape)
y_te = np.concatenate(out_ex_te_1)
print(y_te.shape)

def rul_mdl1(batch_size = None,in_feature = x_tr.shape[2], out_f = 1, len_ts = x_tr.shape[1]):
    inp = layers.Input(batch_shape= (batch_size, len_ts, x_tr.shape[2]), name="Input")
    hid_l1 = layers.LSTM(units = 64, return_sequences = True, name = 'Hidden-Layer-1')(inp)
    hid_l2 = layers.LSTM(units = 64, return_sequences = False, name = 'Hidden-Layer-2')(hid_l1)
    hid_l3 = layers.Dense(units = 8, activation = 'relu', name = 'Hidden-Layer-3')(hid_l2)
    hid_l4 = layers.Dense(units = 8, activation = 'relu', name = 'Hidden-Layer-4')(hid_l3)
    out = layers.Dense(units = 1, activation = 'linear', name = 'Output')(hid_l4)
    M = models.Model(inputs = inp, outputs =  out)
    M.compile(loss='mean_squared_error', optimizer='adam')
    return M
Model1 = rul_mdl1()
Model1.summary()
hist1 = Model1.fit(x_tr, y_tr, epochs=1, batch_size=4096, verbose=1, validation_split=0.25)

def get_fourier_coeff(inp_ar):
    inp_ar_f = []
    for i in range(inp_ar.shape[0]):
        seq_ar = inp_ar[i,:,:]
        seq_f = np.empty([seq_ar.shape[0],seq_ar.shape[1]])
        for j in range(seq_ar.shape[1]):
            seq_t = seq_ar[:,j]
            seq_tf = fft(seq_t)
            seq_tf_sym = seq_tf[0:int(len(seq_tf)/2)]
            seq_tf_sym_real = np.real(seq_tf_sym)
            seq_tf_sym_imag = np.imag(seq_tf_sym)
            seq_tf_sym_f = np.concatenate([seq_tf_sym_real,seq_tf_sym_imag])
            seq_f[:,j] = seq_tf_sym_f
        inp_ar_f.append(seq_f)
    inp_ar_f = np.array(inp_ar_f)
    return inp_ar_f

def get_wavelet_coeff(inp_ar):
    inp_ar_w = []
    for i in range(inp_ar.shape[0]):
        seq_ar = inp_ar[i,:,:]
        seq_w = np.empty([128,seq_ar.shape[1]])
        for j in range(seq_ar.shape[1]):
            seq_t = seq_ar[:,j]
            (ca5,cd5,cd4,cd3,cd2,cd1) = pywt.wavedec(seq_t, 'haar', level = 5)
            w_f = np.concatenate([ca5,cd5,cd4,cd3,cd2,cd1])
#             print(w_f.shape)
            seq_w[:,j] = w_f
        inp_ar_w.append(seq_w)
    inp_ar_w = np.array(inp_ar_w)
    return inp_ar_w

x_tr_cat = x_tr[:,:,0:6]
x_tr_f_num = get_fourier_coeff(x_tr[:,:,6:])
x_tr_f = np.concatenate([x_tr_cat, x_tr_f_num], axis = 2)
x_tr_w_num = get_wavelet_coeff(x_tr[:,:,6:])
x_tr_w = np.concatenate([x_tr_cat, x_tr_w_num], axis = 2)
print(x_tr_w.shape)
x_te_cat = x_te[:,:,0:6]
x_te_f_num = get_fourier_coeff(x_te[:,:,6:])
x_te_f = np.concatenate([x_te_cat, x_te_f_num], axis = 2)
x_te_w_num= get_wavelet_coeff(x_te[:,:,6:])
x_te_w = np.concatenate([x_te_cat, x_te_w_num], axis = 2)
print(x_te_w.shape)
x_te = np.asarray(x_te).astype(np.float32)
x_te_f = np.asarray(x_te_f).astype(np.float32)
x_te_w = np.asarray(x_te_w).astype(np.float32)
y_te = np.asarray(y_te).astype(np.float32)

def rul_mdl2(batch_size = None,in_feature = x_tr_f.shape[2], out_f = 1, len_ts = x_tr_f.shape[1]):
    inp = layers.Input(batch_shape= (batch_size, len_ts, x_tr_f.shape[2]), name="Input")
    hid_l1 = layers.LSTM(units = 64, return_sequences = True, name = 'Hidden-Layer-1')(inp)
    hid_l2 = layers.LSTM(units = 64, return_sequences = False, name = 'Hidden-Layer-2')(hid_l1)
    hid_l3 = layers.Dense(units = 8, activation = 'relu', name = 'Hidden-Layer-3')(hid_l2)
    hid_l4 = layers.Dense(units = 8, activation = 'relu', name = 'Hidden-Layer-4')(hid_l3)
    out = layers.Dense(units = 1, activation = 'linear', name = 'Output')(hid_l4)
    M = models.Model(inputs = inp, outputs =  out)
    M.compile(loss='mean_squared_error', optimizer='adam')
    return M
Model2 = rul_mdl2()
Model2.summary()
hist2 = Model2.fit(x_tr_f, y_tr, epochs=1, batch_size=4096, verbose=1, validation_split=0.25)
   
def rul_mdl3(batch_size = None,in_feature = x_tr_w.shape[2], out_f = 1, len_ts = x_tr_w.shape[1]):
    inp = layers.Input(batch_shape= (batch_size, len_ts, x_tr_w.shape[2]), name="Input")
    hid_l1 = layers.LSTM(units = 64, return_sequences = True, name = 'Hidden-Layer-1')(inp)
    hid_l2 = layers.LSTM(units = 64, return_sequences = False, name = 'Hidden-Layer-2')(hid_l1)
    hid_l3 = layers.Dense(units = 8, activation = 'relu', name = 'Hidden-Layer-3')(hid_l2)
    hid_l4 = layers.Dense(units = 8, activation = 'relu', name = 'Hidden-Layer-4')(hid_l3)
    out = layers.Dense(units = 1, activation = 'linear', name = 'Output')(hid_l4)
    M = models.Model(inputs = inp, outputs =  out)
    M.compile(loss='mean_squared_error', optimizer='adam')
    return M
Model3 = rul_mdl3()
Model3.summary()
hist3 = Model3.fit(x_tr_w, y_tr, epochs=1, batch_size=4096, verbose=1, validation_split=0.25)

def rul_mdl4(batch_size = None,in_feature = x_tr.shape[2], out_f = 1, len_ts = x_tr.shape[1]):
    inp = layers.Input(batch_shape= (batch_size, len_ts, x_tr.shape[2]), name="Input")
    hid_l1 = layers.Bidirectional(layers.LSTM(units = 64, return_sequences = True, name = 'Hidden-Layer-1'))(inp)
    hid_l2 = layers.Bidirectional(layers.LSTM(units = 64, return_sequences = False, name = 'Hidden-Layer-2'))(hid_l1)
    hid_l3 = layers.Dense(units = 8, activation = 'relu', name = 'Hidden-Layer-3')(hid_l2)
    hid_l4 = layers.Dense(units = 8, activation = 'relu', name = 'Hidden-Layer-4')(hid_l3)
    out = layers.Dense(units = 1, activation = 'linear', name = 'Output')(hid_l4)
    M = models.Model(inputs = inp, outputs =  out)
    M.compile(loss='mean_squared_error', optimizer='adam')
    return M
Model4 = rul_mdl4()
Model4.summary()
hist4 = Model4.fit(x_tr, y_tr, epochs=1, batch_size=4096, verbose=1, validation_split=0.25)

def rul_mdl5(batch_size = None,in_feature = x_tr_f.shape[2], out_f = 1, len_ts = x_tr_f.shape[1]):
    inp = layers.Input(batch_shape= (batch_size, len_ts, x_tr_f.shape[2]), name="Input")
    hid_l1 = layers.Bidirectional(layers.LSTM(units = 64, return_sequences = True, name = 'Hidden-Layer-1'))(inp)
    hid_l2 = layers.Bidirectional(layers.LSTM(units = 64, return_sequences = False, name = 'Hidden-Layer-2'))(hid_l1)
    hid_l3 = layers.Dense(units = 8, activation = 'relu', name = 'Hidden-Layer-3')(hid_l2)
    hid_l4 = layers.Dense(units = 8, activation = 'relu', name = 'Hidden-Layer-4')(hid_l3)
    out = layers.Dense(units = 1, activation = 'linear', name = 'Output')(hid_l4)
    M = models.Model(inputs = inp, outputs =  out)
    M.compile(loss='mean_squared_error', optimizer='adam')
    return M
Model5 = rul_mdl5()
Model5.summary()
hist5 = Model5.fit(x_tr_f, y_tr, epochs=1, batch_size=4096, verbose=1, validation_split=0.25)

def rul_mdl6(batch_size = None,in_feature = x_tr_w.shape[2], out_f = 1, len_ts = x_tr_w.shape[1]):
    inp = layers.Input(batch_shape= (batch_size, len_ts, x_tr_w.shape[2]), name="Input")
    hid_l1 = layers.Bidirectional(layers.LSTM(units = 64, return_sequences = True, name = 'Hidden-Layer-1'))(inp)
    hid_l2 = layers.Bidirectional(layers.LSTM(units = 64, return_sequences = False, name = 'Hidden-Layer-2'))(hid_l1)
    hid_l3 = layers.Dense(units = 8, activation = 'relu', name = 'Hidden-Layer-3')(hid_l2)
    hid_l4 = layers.Dense(units = 8, activation = 'relu', name = 'Hidden-Layer-4')(hid_l3)
    out = layers.Dense(units = 1, activation = 'linear', name = 'Output')(hid_l4)
    M = models.Model(inputs = inp, outputs =  out)
    M.compile(loss='mean_squared_error', optimizer='adam')
    return M
Model6 = rul_mdl6()
Model6.summary()
hist6 = Model6.fit(x_tr_w, y_tr, epochs=1, batch_size=4096, verbose=1, validation_split=0.25)

x_tr_cnn = x_tr[:,:,:,np.newaxis]
x_tr_f_cnn = x_tr_f[:,:,:,np.newaxis]
x_tr_w_cnn = x_tr_w[:,:,:,np.newaxis]
print(x_tr_cnn.shape)
print(x_tr_f_cnn.shape)
print(x_tr_w_cnn.shape)
x_te_cnn = x_te[:,:,:,np.newaxis]
x_te_f_cnn = x_te_f[:,:,:,np.newaxis]
x_te_w_cnn = x_te_w[:,:,:,np.newaxis]
print(x_te_cnn.shape)
print(x_te_f_cnn.shape)
print(x_te_w_cnn.shape)

def rul_mdl7(batch_size = None,in_feature = x_tr_cnn.shape[2], out_f = 1, len_ts = x_tr_cnn.shape[1]):
    inp = layers.Input(batch_shape= (batch_size, len_ts, x_tr_cnn.shape[2], x_tr_cnn.shape[3]), name="Input")
    l1 = layers.Conv2D(filters = 3, kernel_size = (x_tr_cnn.shape[1],4), activation = 'relu',name = 'Convolution-1')(inp)
    l2 = layers.MaxPool2D(pool_size=(1, 2), name = 'MaxPool-1')(l1)
    l3 = layers.Conv2D(filters = 3, kernel_size = (1,3), activation = 'relu',name = 'Convolution-2')(l2)
    l4 = layers.MaxPool2D(pool_size=(1, 2), name = 'MaxPool-2')(l3)
    l5 = layers.Flatten(name = 'Flatten')(l4)
    l6 = layers.Dense(units = 8, activation = 'relu',name = 'FC-1')(l5)
    out = layers.Dense(units = 1,activation = 'linear', name = 'output')(l6)
    M = models.Model(inputs = inp, outputs =  out)
    M.compile(loss='mean_squared_error', optimizer='adam')
    return M
Model7 = rul_mdl7()
Model7.summary()
hist7 = Model7.fit(x_tr_cnn, y_tr, epochs=1, batch_size=4096, verbose=1, validation_split=0.25)

def rul_mdl8(batch_size = None,in_feature = x_tr_f_cnn.shape[2], out_f = 1, len_ts = x_tr_f_cnn.shape[1]):
    inp = layers.Input(batch_shape= (batch_size, len_ts, x_tr_f_cnn.shape[2], x_tr_f_cnn.shape[3]), name="Input")
    l1 = layers.Conv2D(filters = 3, kernel_size = (x_tr_f_cnn.shape[1],4), activation = 'relu',name = 'Convolution-1')(inp)
    l2 = layers.MaxPool2D(pool_size=(1, 2), name = 'MaxPool-1')(l1)
    l3 = layers.Conv2D(filters = 3, kernel_size = (1,3), activation = 'relu',name = 'Convolution-2')(l2)
    l4 = layers.MaxPool2D(pool_size=(1, 2), name = 'MaxPool-2')(l3)
    l5 = layers.Flatten(name = 'Flatten')(l4)
    l6 = layers.Dense(units = 8, activation = 'relu',name = 'FC-1')(l5)
    out = layers.Dense(units = 1,activation = 'linear', name = 'output')(l6)
    M = models.Model(inputs = inp, outputs =  out)
    M.compile(loss='mean_squared_error', optimizer='adam')
    return M
Model8 = rul_mdl8()
Model8.summary()
hist8 = Model8.fit(x_tr_f_cnn, y_tr, epochs=1, batch_size=4096, verbose=1, validation_split=0.25)

def rul_mdl9(batch_size = None,in_feature = x_tr_w_cnn.shape[2], out_f = 1, len_ts = x_tr_w_cnn.shape[1]):
    inp = layers.Input(batch_shape= (batch_size, len_ts, x_tr_w_cnn.shape[2], x_tr_w_cnn.shape[3]), name="Input")
    l1 = layers.Conv2D(filters = 3, kernel_size = (x_tr_w_cnn.shape[1],4), activation = 'relu',name = 'Convolution-1')(inp)
    l2 = layers.MaxPool2D(pool_size=(1, 2), name = 'MaxPool-1')(l1)
    l3 = layers.Conv2D(filters = 3, kernel_size = (1,3), activation = 'relu',name = 'Convolution-2')(l2)
    l4 = layers.MaxPool2D(pool_size=(1, 2), name = 'MaxPool-2')(l3)
    l5 = layers.Flatten(name = 'Flatten')(l4)
    l6 = layers.Dense(units = 8, activation = 'relu',name = 'FC-1')(l5)
    out = layers.Dense(units = 1,activation = 'linear', name = 'output')(l6)
    M = models.Model(inputs = inp, outputs =  out)
    M.compile(loss='mean_squared_error', optimizer='adam')
    return M
Model9 = rul_mdl9()
Model9.summary()
hist9 = Model9.fit(x_tr_w_cnn, y_tr, epochs=1, batch_size=4096, verbose=1, validation_split=0.25)

rul_pred_naive_tr = Model1.predict(x_tr,verbose = 1)
rul_pred_fourier_tr = Model2.predict(x_tr_f,verbose = 1)
rul_pred_wavelet_tr = Model3.predict(x_tr_w,verbose = 1)
rul_pred_naive_bilstm_tr = Model4.predict(x_tr,verbose = 1)
rul_pred_fourier_bilstm_tr = Model5.predict(x_tr_f,verbose = 1)
rul_pred_wavelet_bilstm_tr = Model6.predict(x_tr_w,verbose = 1)
rul_pred_naive_cnn_tr = Model7.predict(x_tr_cnn,verbose = 1)
rul_pred_fourier_cnn_tr = Model8.predict(x_tr_f_cnn,verbose = 1)
rul_pred_wavelet_cnn_tr = Model9.predict(x_tr_w_cnn,verbose = 1)
y_tr = y_tr.reshape(-1,1)
res_1 = pd.DataFrame(np.concatenate([rul_pred_naive_tr,rul_pred_fourier_tr,rul_pred_wavelet_tr,rul_pred_naive_bilstm_tr,rul_pred_fourier_bilstm_tr,rul_pred_wavelet_bilstm_tr,rul_pred_naive_cnn_tr,rul_pred_fourier_cnn_tr,rul_pred_wavelet_cnn_tr,y_tr], axis = 1), 
                     columns=['predicted_RUL_naive','predicted_RUL_fourier','predicted_RUL_wavelet','predicted_RUL_Naive_BILSTM','predicted_RUL_fourier_BILSTM','predicted_RUL_wavelet_BILSTM','predicted_RUL_Naive_CNN','predicted_RUL_fourier_CNN','predicted_RUL_wavelet_CNN','Actual_RUL'])
res_1.to_csv('result_rul_0601_tr_FD002.csv')

rul_pred_naive_te = Model1.predict(x_te,verbose = 1)
rul_pred_fourier_te = Model2.predict(x_te_f,verbose = 1)
rul_pred_wavelet_te = Model3.predict(x_te_w,verbose = 1)
rul_pred_naive_bilstm_te = Model4.predict(x_te,verbose = 1)
rul_pred_fourier_bilstm_te = Model5.predict(x_te_f,verbose = 1)
rul_pred_wavelet_bilstm_te = Model6.predict(x_te_w,verbose = 1)
rul_pred_naive_cnn_te = Model7.predict(x_te_cnn,verbose = 1)
rul_pred_fourier_cnn_te = Model8.predict(x_te_f_cnn,verbose = 1)
rul_pred_wavelet_cnn_te = Model9.predict(x_te_w_cnn,verbose = 1)
y_te = y_te.reshape(-1,1)
res_1 = pd.DataFrame(np.concatenate([rul_pred_naive_te,rul_pred_fourier_te,rul_pred_wavelet_te,rul_pred_naive_bilstm_te,rul_pred_fourier_bilstm_te,rul_pred_wavelet_bilstm_te,rul_pred_naive_cnn_te,rul_pred_fourier_cnn_te,rul_pred_wavelet_cnn_te,y_te], axis = 1), 
                     columns=['predicted_RUL_naive','predicted_RUL_fourier','predicted_RUL_wavelet','predicted_RUL_Naive_BILSTM','predicted_RUL_fourier_BILSTM','predicted_RUL_wavelet_BILSTM','predicted_RUL_Naive_CNN','predicted_RUL_fourier_CNN','predicted_RUL_wavelet_CNN','Actual_RUL'])
res_1.to_csv('result_rul_0601_te_FD002.csv')
