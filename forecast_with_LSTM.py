#
#This code pre-processes input dataset to feed to LSTM network, build and LSTM model, train and forecast the output for 2 timesteps ahead 

import sys, os

import pandas as pd
import numpy as np

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import AveragePooling1D


mydf = pd.read_pickle("/home/path/to/input/file.csv")
print(mydf.columns)
mydf.set_index('cmi_ts',drop=True,append=False,inplace=True,verify_integrity=True)
#values = mydf.values
#print(mydf)
print(mydf.columns)
#values = values.astype('float32')
n_vars = 2

lb_steps = 3    ## look back steps
fc_steps = 2    ## forecast steps

cols, names = list(),list()

# go 12 steps back to create input sequence
for i in range(lb_steps,0,-1):
        print("input seq ",i)
        cols.append(mydf.shift(i))
        #names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        names += ['cmi'+'_t-'+str(i)]
        names += ['ghi'+'_t-'+str(i)]
# go 4 steps ahead to create forecast sequence
for i in range(0, fc_steps):
        print("forecast : ",i)
        cols.append(mydf.shift(-i))
        if i == 0:
                #names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
                names += ['cmi_t']
                names += ['ghi_t']
        else:
                #names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
                names += ['cmi'+'_t+'+str(i)]
                names += ['ghi'+'_t+'+str(i)]
print(names)
agg = pd.concat(cols,axis=1)
agg.columns = names
#agg_final = agg[['var2(t-12)','var2(t-11)','var2(t-10)']]
#agg_formatted = agg[['cmi_t-12','cmi_t-11','cmi_t-10','cmi_t-9','cmi_t-8','cmi_t-7','cmi_t-6','cmi_t-5','cmi_t-4','cmi_t-3','cmi_t-2','cmi_t-1','cmi_t','ghi_t','ghi_t+1','ghi_t+2','ghi_t+3']]
#agg_formatted = agg[['ghi_t-12','ghi_t-11','ghi_t-10','ghi_t-9','ghi_t-8','ghi_t-7','ghi_t-6','ghi_t-5','ghi_t-4','ghi_t-3','ghi_t-2','ghi_t-1','ghi_t','ghi_t+1','ghi_t+2','ghi_t+3']]

agg_formatted = agg[['ghi_t-3','ghi_t-2','ghi_t-1','ghi_t','ghi_t+1']]

# get df row count
df_size = agg_formatted.shape[0]

# discard first few and last few rows since they have NaN values for some of the columns
agg_formatted = agg_formatted[lb_steps:df_size - fc_steps + 1]

# slice the pandas dataframe into train & test sets randomly

train = agg_formatted.sample(frac=0.8,random_state=200)
test = agg_formatted.drop(train.index)

train = train.values
test = test.values


# split into inputs and outputs

train_X, train_y = train[:,0:lb_steps],train[:,lb_steps:]
test_X, test_y = test[:,0:lb_steps],test[:,lb_steps:]


#reshape the inputs for LSTM : 3D array with [samples, timesteps, features]

train_X = train_X.reshape((train_X.shape[0], train_X.shape[1],1))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1],1))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

#build the neural networks

model = Sequential()
# add LSTM layer with sample size as 1, time steps "train_X.shape[1]", and # of features 1
model.add(LSTM(50, input_shape=(train_X.shape[1], 1)))
model.add(Dense(2))
model.compile(loss='mae', optimizer='adam')

model.summary()
# fit network
history = model.fit(train_X, train_y, epochs=500, batch_size=2, validation_data=(test_X, test_y), verbose=2, shuffle=False)

#predict using the model

#yhat = model.predict(test_X)
#test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
yhat = model.predict(test_X)

print("forecasting : ", yhat)
print(" actual values : ",test_y)
