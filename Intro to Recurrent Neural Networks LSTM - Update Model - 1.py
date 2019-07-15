import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import keras
import math
from sklearn.metrics import mean_squared_error

# Some functions to help out with
def plot_predictions(test,predicted):
    plt.plot(test, color='red',label='Real IBM Stock Price')
    plt.plot(predicted, color='blue',label='Predicted IBM Stock Price')
    plt.title('IBM Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('IBM Stock Price')
    plt.legend()
    plt.show()

def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))

# First, we get the data
dataset = pd.read_csv('C:/Users/JinCheol/Desktop/BCCSU/JinCheol/Python/data/IBM_2006-01-01_to_2018-01-01.csv', 
                      index_col='Date', 
                      parse_dates=['Date'])
dataset.head()

# Checking for missing values
training_set = dataset[:'2016'].iloc[:,1:2].values
test_set = dataset['2017':].iloc[:,1:2].values



# We have chosen 'High' attribute for prices. Let's see what it looks like
dataset["High"][:'2016'].plot(figsize=(16,4),legend=True)
dataset["High"]['2017':].plot(figsize=(16,4),legend=True)
plt.legend(['Training set (Before 2017)','Test set (2017 and beyond)'])
plt.title('IBM stock price')
plt.show()


# Scaling the training set
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)


# Since LSTMs store long term memory state, we create a data structure with 90 timesteps and 1 output
# So for each element of training set, we have 90 previous training set elements 
X_train = []
y_train = []
rang=90

for i in range(rang, training_set_scaled.shape[0]):
    X_train.append(training_set_scaled[i-rang:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping X_train for efficient modelling
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# The LSTM architecture
regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
# Second LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Third LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Fourth LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
# The output layer
regressor.add(Dense(units=1))

# ===================== Another way to build layers ===========================
# model = keras.Sequential([
#     # First LSTM layer with Dropout regularisation
#     keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)),
#     keras.layers.Dropout(0.2),
#     keras.layers.LSTM(units=50, return_sequences=True),
#     keras.layers.Dropout(0.2),
#     keras.layers.LSTM(units=50, return_sequences=True),
#     keras.layers.Dropout(0.2),
#     
#     # The output layer
#     keras.layers.Dense(units=1)
# ])
# =============================================================================


# Compiling the RNN
regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
# Fitting to the training set
regressor.fit(X_train,y_train, epochs=50, batch_size=32)


# Preparing X_test and predicting the prices
t=test_set.shape[0]
X_test = [] 
y_test = []
for i in range(0, t):
    if(i==0):
        # Take the data of last 90 points in the training data as the initial input
        Temp_X_test=X_train[X_train.shape[0]-1, :, :]
        # Conver shape to be fit in the model
        Temp_X_test = np.array(Temp_X_test)
        Temp_X_test = np.reshape(Temp_X_test, (1, rang, 1))
        # Predict the next price & update y_test
        pred_y=regressor.predict(Temp_X_test)
        y_test.append(pred_y)
        # update Temp_X_test
        Temp_X_test=np.concatenate([Temp_X_test[0][1:rang], y_test[i]])
        # update X_test
        X_test.append(Temp_X_test)        
        
        #*****************
        # update the model
        #*****************
        # using only the most time window of recent 90 points where the last time point is predicted value
        regressor.fit(np.reshape(Temp_X_test, (1, rang, 1)), np.reshape(pred_y, (1)), epochs=5, verbose=0)
        # using accumulative time windows in test data
        #regressor.fit(np.reshape(np.array(X_test), (i+1, rang, 1)) , np.reshape(y_test, (i+1)), epochs=5, verbose=0)
        # using all time windows from train data and accumulative time windows in test data
        #regressor.fit(np.concatenate([X_train, X_test]) , np.concatenate([y_train, np.reshape(y_test, (i+1))]), epochs=5, batch_size=32, verbose=0)
                
    elif(i>0):
        # Conver shape to be fit in the model
        Temp_X_test = np.array(Temp_X_test)
        Temp_X_test = np.reshape(Temp_X_test, (1, rang, 1))
        # Predict the next price
        pred_y=regressor.predict(Temp_X_test)
        y_test.append(pred_y)
        # update Temp_X_test
        Temp_X_test=np.concatenate([Temp_X_test[0][1:rang], y_test[i]])             
        # update X_test
        X_test.append(Temp_X_test)
        
        #*****************
        # update the model
        #*****************
        # using only the most time window of recent 90 points where the last time point is predicted value
        regressor.fit(np.reshape(Temp_X_test, (1, rang, 1)), np.reshape(pred_y, (1)), epochs=5, verbose=0)
        # using accumulative time windows in test data
        #regressor.fit(np.reshape(np.array(X_test), (i+1, rang, 1)), np.reshape(y_test, (i+1)), epochs=5, verbose=0)
        # using all time windows from train data and accumulative time windows in test data
        #regressor.fit(np.concatenate([X_train, X_test]) , np.concatenate([y_train, np.reshape(y_test, (i+1))]), epochs=5, verbose=0)
        
    # print the progress
    if(i==round(t*1/4)):
        print("Progress : {}%".format(25))
    elif(i==round(t*2/4)):
        print("Progress : {}%".format(50))
    elif(i==round(t*3/4)):
        print("Progress : {}%".format(75))
    elif(i==round(t*4/4)):
        print("Progress : {}%".format(100))

#
X_test = np.array(X_test)
X_test = np.reshape(X_test, (t, rang, 1))
y_test = np.array(y_test)
y_test = np.reshape(y_test, (t, 1))
# convert y back to the original unit
y_test = sc.inverse_transform(y_test)

# Visualizing the results for LSTM
plot_predictions(test_set, y_test)

# Evaluating our model
return_rmse(test_set, y_test)

# save the session
import dill                            #pip install dill --user
result_dir='/Users/JinCheol/Desktop/BCCSU/JinCheol/Python/Tensorflow_Tutorial/LSTM/'
dill.dump_session(result_dir+'Updated_Model-1.pkl')
# load the session again:
#dill.load_session(result_dir+'Updated_Model-1.pkl')

