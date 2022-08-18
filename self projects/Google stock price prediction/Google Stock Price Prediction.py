#!/usr/bin/env python
# coding: utf-8

# In[123]:


import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


# In[124]:



from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(10)


# In[125]:


df =pd.read_csv("prices.csv", header=0)
display (df)


# In[126]:


print(df.shape)


# In[127]:


print(df.columns)


# In[128]:


df.symbol.value_counts()


# In[129]:


df.symbol.unique()


# In[130]:


display(df.symbol.unique().shape)


# In[131]:


df.symbol.unique()[0:20]


# In[132]:


print(len(df.symbol.values))


# In[133]:


df.info()


# In[134]:


df.describe()


# In[135]:


df.isnull().sum()


# In[136]:


df.date.unique()


# In[137]:


pd.DataFrame(df.date.unique())


# In[138]:


df.duplicated().sum()


# In[139]:


#Calling the file in nyse named securities.csv, It has the company details 
comp_info = pd.read_csv('securities.csv')
comp_info


# In[140]:


comp_info["Ticker symbol"].nunique()


# In[141]:


comp_info.info()


# In[142]:


comp_info.isnull().sum()


# In[143]:


comp_info.describe()


# In[144]:


comp_info.loc[comp_info.Security.str.startswith('Face') , :]


# In[145]:


comp_info.loc[comp_info.Security.str.startswith('Acc') , :]


# In[146]:


comp_plot = comp_info.loc[(comp_info["Security"] == 'Yahoo Inc.') | (comp_info["Security"] == 'Xerox Corp.') | (comp_info["Security"] == 'Adobe Systems Inc')
              | (comp_info["Security"] == 'Microsoft Corp.') | (comp_info["Security"] == 'Adobe Systems Inc') 
              | (comp_info["Security"] == 'Facebook') | (comp_info["Security"] == 'Goldman Sachs Group') , ["Ticker symbol"] ]["Ticker symbol"] 
print(comp_plot)


# In[147]:


for i in comp_plot:
    print (i)


# In[148]:


def plotter(code):
    # Function used to create graphs for 6 companies 
    global closing_stock ,opening_stock
    #creating plot of all 6 company for opening and closing stock  total 12 graphs
    # Below statement create 2X2 empty chart 
    f, axs = plt.subplots(2,2,figsize=(15,8))
    # total 12 graphs
    # creating plot opening prize of particular company
    plt.subplot(212)
    #taking name of the company as code, get all records related to one company
    company = df[df['symbol']==code]
    #taking the values of one company and taking its open column values to 1D array
    company = company.open.values.astype('float32')
    #reshaping the open stock value from 1D  to 2D .
    company = company.reshape(-1, 1)
    # putting the value of company in opening_stock 
    opening_stock = company
    # plotting the data with green graph between "Time" and "prices vs time"
    
    plt.grid(True)# enalbling the grid in graph
    plt.xlabel('Time') # setting X axis as time
    # setting Y axis as company name + open stock prices
    plt.ylabel(code + " open stock prices") 
    plt.title('prices Vs Time') # setting title
    plt.plot(company , 'g') # calling the graph with green graph line
    
    # creating plot closing prize of particular company
    plt.subplot(211)
    #taking name of the company as code
    company_close = df[df['symbol']==code]
    #taking the values of one company and taking its close column values
    company_close = company_close.close.values.astype('float32')
    #reshaping the open column value in 1D and calling it closing_stock
   # -1 for unknown dimension
    company_close = company_close.reshape(-1, 1)
    # putting company_close value in closing_stock 
    closing_stock = company_close
    # plotting the data graph between "Time" and "prices vs time"
    plt.xlabel('Time') # setting x axis as time
    plt.ylabel(code + " close stock prices")# setting y axis as company name + open stock prices
    plt.title('prices Vs Time') # setting title as price vs time
    plt.grid(True) # enabling the grid in graph
    plt.plot(company_close , 'b') #creating the data graph in blue graph line
    plt.show() # calling the graph


# In[149]:


# for i in comp_plot:
#     plotter(i)


# In[150]:


stocks= np.array (df[df.symbol.isin (['GOOGL'])].close)
print(stocks)


# In[151]:


display (stocks.shape)


# In[152]:


stocks = stocks.reshape(len(stocks) , 1)
print (stocks.shape)
print(stocks)


# In[153]:


from sklearn.preprocessing import MinMaxScaler
#scaling features between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1)) 
stocks = scaler.fit_transform(stocks) 
display (stocks)


# In[154]:


print (stocks.shape) 


# In[155]:


train = int(len(stocks) * 0.80)
print (train)


# In[156]:


test = len(stocks) - train 
print (test)


# In[157]:


train = stocks[0:train]
display (train.shape)
print(train)


# In[158]:


test = stocks[len(train) : ]
display(test.shape)
display (test)


# In[159]:


#creating function to create trainX,testX and target(trainY, testY)
def process_data(data , n_features):
    dataX, dataY = [], [] # creating data for dataset and dividing inta X,Y
    for i in range(len(data)-n_features):
        # taking i range from total size- 3 
        a = data[i:(i+n_features), 0]
        # Here a is value of data from i to i+ n_features, ie two values and put it in dataX 
        dataX.append(a) #putting a in dataX
        #here dataY takes the value of data of i + n_features
        dataY.append(data[i + n_features, 0])
        # putting i+ n_features in  dataY
    return np.array(dataX), np.array(dataY)
# returning dataX and dataY in array



# In[160]:


n_features = 30
# Here we create train X, Train Y and test X, Test Y data where trainX, testX has two value is each block

trainX, trainY = process_data(train, n_features)
print(trainX.shape , trainY.shape)


# In[161]:


testX, testY = process_data(test, n_features)
print (testX.shape , testY.shape)


# In[162]:


stocksX, stocksY = process_data(stocks, n_features)
print (stocksX.shape , stocksY.shape)


# In[163]:


display (trainX[:10])


# In[164]:


display (trainY[:10])


# In[165]:


# reshaping trainX and testX to use in deep learning model
trainX = trainX.reshape(trainX.shape[0] , 1 ,trainX.shape[1])
display (trainX.shape)


# In[166]:


testX = testX.reshape(testX.shape[0] , 1 ,testX.shape[1])
display (testX.shape)


# In[167]:


stocksX= stocksX.reshape(stocksX.shape[0] , 1 ,stocksX.shape[1])
display (stocksX.shape)


# In[168]:


# helps us do mathematical operations
import math 
# for setting layers one by one neural layer in model 
import tensorflow
from tensorflow import keras
from keras.models import Sequential 
# types of layers
from keras.layers import Dense , BatchNormalization , Dropout , Activation 
# types of RNN
from keras.layers import LSTM , GRU 
#It puts the data in between given range to set data before putting layer
from sklearn.preprocessing import MinMaxScaler 
# In this method the errors in column is squared and then mean is found 
from sklearn.metrics import mean_squared_error 
# Optimizers used
from keras.optimizers import Adam , SGD , RMSprop


# In[169]:


#Checkpointing the model when required and using other call-backs.
filepath="stock_weights1.hdf5"
from keras.callbacks import ReduceLROnPlateau , ModelCheckpoint
# ReduceLROnPlateau- This reduce the learning rate when the matrix stop improving or  too close to reduce overfitting
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=0.0001, patience=1, verbose=1)
#This check point will stop processing, if the model is not improving.
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')


# In[170]:


# creating model for training data using sequential to give series wise output between layers
model = Sequential()
# GRU or Gated Recurrent Unit used for matrix manipulation within Recurrent layer
#This is the input Layer 
model.add(LSTM(64 , input_shape = (1 , n_features) , return_sequences=True))
#dropout is used to remove overfitting data on each layer of neural network
model.add(Dropout(0.4))
#Long Short Term Memory is a type of RNN specially used for time series problems
model.add(LSTM(128,return_sequences=True))
#dropout is used to remove overfitting data on each layer of neural network
model.add(Dropout(0.4))
model.add(LSTM(128))
model.add(Dropout(0.2))

#Dense layer are fully connected neural networks 
#model.add(Dense(64 ,  activation = 'relu'))
#This is the output Layer, Output is only one neuron 
model.add(Dense(1))
#for getting the details of our models
print(model.summary())


# In[171]:


# Selecting the loss measurement metrics and optimizer for our model, to find out mean square error
model.compile(loss='mean_squared_error', optimizer=Adam(lr = 0.0005) , metrics = ['mean_squared_error'])


# In[172]:


# fitting the data i.e training the trainX, to relate to trainY
# epochs is the times each data in send to fit
# batch size is the size of information send at a time
# validation_data is the validation or data used for testing 
history = model.fit(trainX, trainY, epochs=100 , batch_size = 128 , 
          callbacks = [checkpoint , lr_reduce] , validation_data = (testX,testY))    


# In[173]:



test_pred = model.predict(testX)
display (test_pred [:10])


# In[174]:


test_pred = scaler.inverse_transform(test_pred)
display (test_pred [:10])


# In[175]:


testY = testY.reshape(testY.shape[0] , 1)
#Converting reshaped list in 1D array so that it will be efficient in plotting
testY = scaler.inverse_transform(testY)
# taking testY from 1 to 10
display (testY[:10])


# In[176]:


from sklearn.metrics import r2_score


# In[177]:

plt.figure(0)
# Ploting the graph of stock prices with time
print("Red - Predicted Stock Prices  ,  Blue - Actual Stock Prices")
plt.rcParams["figure.figsize"] = (15,7)
# testY is the blue line
plt.plot(testY , 'b')
# pred is the red line
plt.plot(test_pred , 'r')
# Setting x axis as time
plt.xlabel('Time')
# Setting y axis as stock prices
plt.ylabel('Stock Prices')
# setting title 
plt.title('Check the accuracy of the model with time')
# enabling grids in graph 
plt.grid(True)
# it call the graph with labels, titles, lines
plt.show()


# In[178]:


train_pred = model.predict(trainX)
train_pred = scaler.inverse_transform(train_pred)
trainY = trainY.reshape(trainY.shape[0] , 1)
trainY = scaler.inverse_transform(trainY)
print ('Display Accuracy Training Data')
display (r2_score(trainY,train_pred))


# In[179]:

plt.figure(1)
# Ploting the graph of stock prices with time - Training Data
print("Red - Predicted Stock Prices  ,  Blue - Actual Stock Prices")
plt.rcParams["figure.figsize"] = (15,7)
plt.plot(trainY  , 'b')
plt.plot(train_pred, 'r')
plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.title('Check the accuracy of the model with time')
plt.grid(True)
plt.show()


# In[180]:


stocks_pred = model.predict(stocksX)
stocks_pred = scaler.inverse_transform(stocks_pred)
stocksY = stocksY.reshape(stocksY.shape[0] , 1)
stocksY = scaler.inverse_transform(stocksY)
print ('Display Accuracy Training Data')
display (r2_score(stocksY,stocks_pred))


# In[181]:

plt.figure(2)
plt.rcParams["figure.figsize"] = (15,7)
plt.plot(stocksY  , 'b')
plt.plot(stocks_pred, 'r')
plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.title('Check the accuracy of the model with time')
plt.grid(True)
plt.show()


# In[182]:


from sklearn.metrics import mean_squared_error
print(mean_squared_error(stocksY,stocks_pred))
print(mean_squared_error(trainY,train_pred))
print(mean_squared_error(testY,test_pred))


# In[183]:


print(r2_score(stocksY,stocks_pred))
print(r2_score(trainY,train_pred))
print(r2_score(testY,test_pred))

