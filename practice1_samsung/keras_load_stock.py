import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Conv1D, MaxPooling1D, Flatten, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn. preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


######### 1. 데이터
samsung_x=np.load('./data/samsung_x.npy')
samsung_x_predict=np.load('./data/samsung_x_predict.npy')
samsung_y=np.load('./data/samsung_y.npy')
bit_x=np.load('./data/bit_x.npy')
bit_x_predict=np.load('./data/bit_x_predict.npy')


# train, test 분리
samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test=train_test_split(samsung_x, samsung_y, train_size=0.8)
bit_x_train, bit_x_test=train_test_split(bit_x, train_size=0.8)

# print(samsung_x.shape) #(621, 5, 4)
# print(samsung_y.shape) #(621, 1)
# print(bit_x.shape) #(621, 5, 5)

samsung_x_predict=samsung_x_predict.reshape(1,5,4)
bit_x_predict=bit_x_predict.reshape(1,5,5)

######### 2. LSTM 회귀모델
model = load_model('./model/samsung-693-745744.5000.hdf5')


#4. 평가, 예측
loss=model.evaluate([samsung_x_test, bit_x_test], samsung_y_test, batch_size=1000)
samsung_y_predict=model.predict([samsung_x_predict, bit_x_predict])

print("loss : ", loss)
print("2020.11.20. 삼성전자 종가 :" , samsung_y_predict)