# 네개의 데이터 앙상블 
# 컬럼 수 각각 3, 4, 5, 6
# 월요일 삼성 09시 시가(시작가) / 마감 : 일요일 23시 59분 59초
# 비트와 삼성은 20일 데이터 쓰지 말고 금과 코스닥은 20일 데이터 쓸 것

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Conv1D, MaxPooling1D, Flatten, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn. preprocessing import StandardScaler, MinMaxScaler
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
samsung=samsung[['시가', '고가', '저가', '개인', '종가']]
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


samsung_x=samsung[['고가', '저가', '개인','종가']]
samsung_y=samsung[['시가']]

# 11월 20일 데이터 삭제
samsung_x.drop(samsung_x.index[-1], inplace=True)
bit.drop(bit.index[-1], inplace=True)
samsung_y.drop(samsung_y.index[-1], inplace=True)


#to numpy
samsung_x=samsung_x.to_numpy()
samsung_y=samsung_y.to_numpy()
bit_x=bit.to_numpy()
gold_x=gold.to_numpy()
kosdak_x=kosdak.to_numpy()



#데이터 스케일링
scaler1=StandardScaler()
scaler1.fit(samsung_x)
samsung_x=scaler1.transform(samsung_x)

scaler2=StandardScaler()
scaler2.fit(bit_x)
bit_x=scaler2.transform(bit_x)

scaler3=StandardScaler()
scaler3.fit(gold_x)
gold_x=scaler3.transform(gold_x)

scaler4=MinMaxScaler()
scaler4.fit(kosdak_x)
kosdak_x=scaler4.transform(kosdak_x)

# x 데이터 다섯개씩 자르기
def split_data(x, size) :
    data=[]
    for i in range(x.shape[0]-size+1) :
        data.append(x[i:i+size,:])
    return np.array(data)

size=5
samsung_x=split_data(samsung_x, size)
bit_x=split_data(bit_x, size)
gold_x=split_data(gold_x, size)
kosdak_x=split_data(kosdak_x, size)
bit_x=bit_x[:samsung_x.shape[0],:]
gold_x=gold_x[:samsung_x.shape[0],:]
kosdak_x=kosdak_x[:samsung_x.shape[0],:]

# y 데이터 추출
samsung_y=samsung_y[size+1:, :]



# predict 데이터 추출
samsung_x_predict=samsung_x[-1]
bit_x_predict=bit_x[-1]
gold_x_predict=gold_x[-1]
kosdak_x_predict=kosdak_x[-1]

samsung_x=samsung_x[:-2, :, :]
bit_x=bit_x[:-2, :, :]
gold_x=gold_x[:-2, :, :]
kosdak_x=kosdak_x[:-2, :, :]

print(samsung_x.shape) # (620, 5, 4)
print(bit_x.shape) #(620, 5, 5)
print(gold_x.shape) #(620, 5, 6)
print(kosdak_x.shape) #(620, 5, 3)
print(samsung_y.shape) # (620, 1)

samsung_x=samsung_x.astype('float32')
samsung_y=samsung_y.astype('float32')
samsung_x_predict=samsung_x_predict.astype('float32')
bit_x=bit_x.astype('float32')
bit_x_predict=bit_x_predict.astype('float32')
gold_x=gold_x.astype('float32')
gold_x_predict=gold_x_predict.astype('float32')

np.save('./data/monday/samsung_x.npy', arr=samsung_x)
np.save('./data/monday/samsung_x_predict.npy', arr=samsung_x_predict)
np.save('./data/monday/samsung_y.npy', arr=samsung_y)
np.save('./data/monday/bit_x.npy', arr=bit_x)
np.save('./data/monday/bit_x_predict.npy', arr=bit_x_predict)
np.save('./data/monday/gold_x.npy', arr=gold_x)
np.save('./data/monday/gold_x_predict.npy', arr=gold_x_predict)
np.save('./data/monday/kosdak_x.npy', arr=kosdak_x)
np.save('./data/monday/kosdak_x_predict.npy', arr=kosdak_x_predict)

# train, test 분리
samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test=train_test_split(samsung_x, samsung_y, train_size=0.8)
bit_x_train, bit_x_test, gold_x_train, gold_x_test, kosdak_x_train, kosdak_x_test=train_test_split(bit_x, gold_x, kosdak_x, train_size=0.8)


samsung_x_predict=samsung_x_predict.reshape(1,5,4)
bit_x_predict=bit_x_predict.reshape(1,5,5)
gold_x_predict=gold_x_predict.reshape(1,5,6)
kosdak_x_predict=kosdak_x_predict.reshape(1,5,3)



######### 2. LSTM 회귀모델
samsung_input1=Input(shape=(5,4))
samsung_layer1=LSTM(512, activation='relu')(samsung_input1)
samsunt_layer1=Dropout(0.2)(samsung_layer1)
samsung_layer2=Dense(4096, activation='relu')(samsung_layer1)
samsung_layer3=Dropout(0.2)(samsung_layer2)
samsung_layer3=Dense(2048, activation='relu')(samsung_layer3)
samsung_layer3=Dropout(0.2)(samsung_layer3)
samsung_layer3=Dense(512, activation='relu')(samsung_layer3)
samsung_layer3=Dense(256, activation='relu')(samsung_layer3)
samsung_layer3=Dense(128, activation='relu')(samsung_layer3)
samsung_layer3=Dense(64, activation='relu')(samsung_layer3)
samsung_output=Dense(1)(samsung_layer3)



bit_input1=Input(shape=(5,5))
bit_layer1=LSTM(300, activation='relu')(bit_input1)
bit_layer4=Dense(500,activation='relu')(bit_layer1)
bit_layer4=Dropout(0.1)(bit_layer4)
bit_layer5=Dense(200,activation='relu')(bit_layer4)
bit_layer6=Dense(100,activation='relu')(bit_layer5)
bit_layer7=Dense(50,activation='relu')(bit_layer6)
bit_output=Dense(1)(bit_layer7)

gold_input1=Input(shape=(5,6))
gold_layer1=LSTM(200, activation='relu')(gold_input1)
gold_layer4=Dense(700,activation='relu')(gold_layer1)
goid_layer4=Dropout(0.1)(gold_layer4)
gold_layer5=Dense(200,activation='relu')(gold_layer4)
gold_layer6=Dense(100,activation='relu')(gold_layer5)
gold_layer7=Dense(50,activation='relu')(gold_layer6)
gold_output=Dense(1)(gold_layer7)


kosdak_input1=Input(shape=(5,3))
kosdak_layer1=LSTM(400, activation='relu')(kosdak_input1)
kosdak_layer2=Dense(900,activation='relu')(kosdak_layer1)
kosdak_layer2=Dropout(0.1)(kosdak_layer2)
kosdak_layer4=Dense(200,activation='relu')(kosdak_layer2)
kosdak_layer5=Dense(20,activation='relu')(kosdak_layer4)
kosdak_layer6=Dense(10,activation='relu')(kosdak_layer5)
kosdak_layer7=Dense(5,activation='relu')(kosdak_layer6)
kosdak_output=Dense(1)(bit_layer7)



merge1=concatenate([samsung_output, bit_output, gold_output, kosdak_output])

output1=Dense(5000)(merge1)
output1=Dropout(0.1)(output1)
output2=Dense(3000)(output1)
output2=Dropout(0.1)(output2)
output3=Dense(800)(output2)
output4=Dense(30)(output3)
output5=Dense(1)(output4)

model=Model(inputs=[samsung_input1, bit_input1, gold_input1, kosdak_input1], outputs=output5)

model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es=EarlyStopping(monitor='val_loss',  patience=100, mode='auto')
modelpath='./model/samsung-monday-{epoch:02d}-{val_loss:.4f}.hdf5'
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit([samsung_x_train, bit_x_train, gold_x_train, kosdak_x_train], samsung_y_train, epochs=10000, batch_size=100, validation_split=0.2, callbacks=[es, cp])


#4. 평가, 예측
loss=model.evaluate([samsung_x_test, bit_x_test, gold_x_test, kosdak_x_test], samsung_y_test, batch_size=100)
samsung_y_predict=model.predict([samsung_x_predict, bit_x_predict, gold_x_predict, kosdak_x_predict])

print("loss : ", loss)
print("2020.11.23. 월요일 삼성전자 시가 :" , samsung_y_predict)