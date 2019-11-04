# House-Sale-Price-Prediction
  房價預測 (回歸模型)
  
任務:

• 用train.csv跟valid.csv訓練模型（一行是一筆房屋交易資料的紀錄，包括id, price與21種房屋參數）

• 將test.csv中的每一筆房屋參數，輸入訓練好的模型，預測其房價

• 將預測結果上傳到Kaggle（從“Submit Predictions”連結）

• 看系統幫你算出來的Mean Abslute Error（MAE，就是跟實際房價差多少，取絕對值）分數夠不夠好？

• 嘗試改進預測模型

工作環境:

python3.5
tensorflow
keras

程式流程:

1.匯入模組

import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers import *

from sklearn.preprocessing import *

2.讀取檔案

data_train = pd.read_csv('train-v3.csv')

x_train = data_train.drop(['price','id'],axis=1).values  

y_train = data_train['price'].values  # read price column

data_valid = pd.read_csv('valid-v3.csv')

x_valid = data_valid.drop(['price','id'],axis=1).values

y_valid = data_valid['price'].values

data_test = pd.read_csv('test-v3.csv')

x_test = data_test.drop('id',axis=1).values

3.數據預處理

x_train = scale(x_train)

x_valid = scale(x_valid)

x_test = scale(x_test)

x_train = preprocessing.scale(x_train)

x_valid = preprocessing.scale(x_valid)

4.訓練模型

model = Sequential()
model.add(Dense(units=512,input_dim=x_train.shape[1],kernel_initializer='normal',activation='relu'))
model.add(Dense(units=1024,kernel_initializer='normal',activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=512,kernel_initializer='normal',activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=1024,kernel_initializer='normal',activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=x_train.shape[1],input_dim=32,kernel_initializer='normal',activation='relu'))
model.add(Dense(1,kernel_initializer='normal'))
print(model.summary())
model.compile(loss='MAE',optimizer='adam')
model.fit(x_train,y_train,batch_size=64,epochs=80,validation_data=(x_valid,y_valid))

5.儲存

Y_predict = model.predict(x_test)

np.savetxt('test.csv',Y_predict,delimiter=',')

結果分析:

使用預測出來的valid 和原來的答案做比較，分析與答案差比較多之數據與幾乎猜對之數據，
如果部分數據大於標準值，可能會導致預測錯誤，此外grade也會因為與數據與其他有所差別可能也會導致預測錯誤

優點:若數據與其他數據沒有太大的變動，可以預測準確

缺點:若數據與其他數據有太大變動可能會預測錯誤

改善:或許可嘗試多丟幾筆數字相差較大的資料去訓練
