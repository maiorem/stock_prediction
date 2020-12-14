# 앙상블 없이 하나의 데이터로 합쳐서 LSTM 사용하기

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn. preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

samsung=pd.read_csv('./data/csv/삼성전자 1120.csv', engine='python', header=0, index_col=0, sep=',')
bit=pd.read_csv('./data/csv/비트컴퓨터 1120.csv',  engine='python', header=0, index_col=0, sep=',')
gold=pd.read_csv('./data/csv/금현물.csv',  engine='python', header=0, index_col=0, sep=',')
kosdak=pd.read_csv('./data/csv/코스닥.csv',  engine='python', header=0, index_col=0, sep=',')



#정렬을 일자별 오름차순으로 변경
samsung=samsung.sort_values(['일자'], ascending=['True'])
bit=bit.sort_values(['일자'], ascending=['True'])
gold=gold.sort_values(['일자'], ascending=['True'])
kosdak=kosdak.sort_values(['일자'], ascending=['True'])

# print(gold) # 시가 고가 저가 종가 거래량 거래대금(백만)
# print(kosdak) # 시가 고가 저가 (float)

#필요한 컬럼만
samsung=samsung[['시가', '고가', '저가', '종가']]
bit=bit[['시가', '고가', '저가', '개인', '종가']]
gold=gold[['시가', '고가', '저가', '종가', '거래량', '거래대금(백만)']]
kosdak=kosdak[['시가', '저가', '고가']]


#콤마 제거 후 문자를 정수로 변환
for i in range(len(samsung.index)) :
    for j in range(len(samsung.iloc[i])) :
        samsung.iloc[i, j]=int(samsung.iloc[i, j].replace(',', ''))

for i in range(len(bit.index)) :
    for j in range(len(bit.iloc[i])) :
        bit.iloc[i, j]=int(bit.iloc[i, j].replace(',', ''))

for i in range(len(gold.index)) :
    for j in range(len(gold.iloc[i])) :
        gold.iloc[i, j]=int(gold.iloc[i, j].replace(',', ''))

# 싯가 2000000 이상의 일자 데이터 삭제
two_million=samsung[samsung['시가']>=2000000].index
samsung.drop(two_million, inplace=True)


samsung_x=samsung[['시가','고가', '저가', '종가']]
samsung_y=samsung[['시가']]

# 11월 20일 데이터 삭제
samsung_x.drop(samsung_x.index[-1], inplace=True)
bit.drop(bit.index[-1], inplace=True)
gold.drop(gold.index[-1], inplace=True)
kosdak.drop(kosdak.index[-1], inplace=True)
samsung_y.drop(samsung_y.index[-1], inplace=True)

#to numpy
samsung_x=samsung_x.to_numpy()
samsung_y=samsung_y.to_numpy()
bit_x=bit.to_numpy()
gold_x=gold.to_numpy()
kosdak_x=kosdak.to_numpy()


# 데이터 행 맞추기
bit_x=bit_x[:samsung_x.shape[0],:]
gold_x=gold_x[:samsung_x.shape[0],:]
kosdak_x=kosdak_x[:samsung_x.shape[0],:]



# print(samsung_x.shape)
# print(bit_x.shape)
# print(gold_x.shape)
# print(kosdak_x.shape)

# (626, 4)
# (626, 5)
# (626, 6)
# (626, 3)


# 데이터 2차원으로 합치기
big_x=np.concatenate([samsung_x, bit_x, gold_x, kosdak_x], axis=1)
# print(big_x.shape) #(626, 18)


#데이터 스케일링
scaler=StandardScaler()
scaler.fit(big_x)
big_x=scaler.transform(big_x)


# x 데이터 다섯개씩 자르기
def split_data(x, size) :
    data=[]
    for i in range(x.shape[0]-size+1) :
        data.append(x[i:i+size,:])
    return np.array(data)

size=5
big_x=split_data(big_x, size)

# y 데이터 추출
big_y=samsung_y[size+1:, :]


# predict 데이터 추출
big_x_predict=big_x[-1]
big_x=big_x[:-2, :, :]

print(big_x.shape) #(620, 5, 18)
print(big_x_predict.shape) #(5, 18)
print(big_y.shape) #(620, 1)

big_x=big_x.astype('float32')
big_y=big_y.astype('float32')
big_x_predict=big_x_predict.astype('float32').reshape(1, big_x_predict.shape[0], big_x_predict.shape[1])


np.save('./data/monday/lstm_big_x.npy', arr=big_x)
np.save('./data/monday/lstm_big_x_predict.npy', arr=big_x_predict)
np.save('./data/monday/lstm_big_y.npy', arr=big_y)

# train, test 분리
big_x_train, big_x_test, big_y_train, big_y_test=train_test_split(big_x, big_y, train_size=0.8)


######### 2. LSTM 회귀모델
model=Sequential()
model.add(LSTM(80, activation='relu', input_shape=(5, 18)))
model.add(Dropout(0.1))
model.add(Dense(5000, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(900, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))

model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es=EarlyStopping(monitor='val_loss',  patience=100, mode='auto')
modelpath='./model/samsung-noensemble-lstm-{epoch:02d}-{val_loss:.4f}.hdf5'
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(big_x_train, big_y_train, epochs=10000, batch_size=100, validation_split=0.2, callbacks=[es, cp])


#4. 평가, 예측
loss=model.evaluate(big_x_test, big_y_test, batch_size=100)
samsung_y_predict=model.predict(big_x_predict)

print("loss : ", loss)
print("2020.11.23. 월요일 삼성전자 시가 :" , samsung_y_predict)
