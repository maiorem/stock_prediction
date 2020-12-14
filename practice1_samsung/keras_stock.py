import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Conv1D, MaxPooling1D, Flatten, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn. preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


samsung=pd.read_csv('./data/csv/삼성전자 1120.csv', engine='python', header=0, index_col=0, sep=',')
bit=pd.read_csv('./data/csv/비트컴퓨터 1120.csv',  engine='python', header=0, index_col=0, sep=',')


#정렬을 일자별 오름차순으로 변경
samsung=samsung.sort_values(['일자'], ascending=['True'])
bit=bit.sort_values(['일자'], ascending=['True'])



#필요한 컬럼만
samsung=samsung[['시가', '고가', '저가', '개인', '종가']]
bit=bit[['시가', '고가', '저가', '개인', '종가']]



#콤마 제거 후 문자를 정수로 변환
for i in range(len(samsung.index)) :
    for j in range(len(samsung.iloc[i])) :
        samsung.iloc[i, j]=int(samsung.iloc[i, j].replace(',', ''))

for k in range(len(bit.index)) :
    for l in range(len(bit.iloc[k])) :
        bit.iloc[k, l]=int(bit.iloc[k, l].replace(',', ''))


# 싯가 2000000 이상의 일자 데이터 삭제
two_million=samsung[samsung['시가']>=2000000].index
samsung.drop(two_million, inplace=True)


samsung_x=samsung[['시가', '고가', '저가','개인']]
samsung_y=samsung[['종가']]

# 11월 20일 데이터 삭제
samsung_x.drop(samsung_x.index[-1], inplace=True)
bit.drop(bit.index[-1], inplace=True)
samsung_y.drop(samsung_y.index[-1], inplace=True)


#to numpy
samsung_x=samsung_x.to_numpy()
bit_x=bit.to_numpy()
samsung_y=samsung_y.to_numpy()

#데이터 스케일링
scaler1=StandardScaler()
scaler1.fit(samsung_x)
samsung_x=scaler1.transform(samsung_x)

scaler2=StandardScaler()
scaler2.fit(bit_x)
bit_x=scaler2.transform(bit_x)


# x 데이터 다섯개씩 자르기
def split_data(x, size) :
    data=[]
    for i in range(x.shape[0]-size+1) :
        data.append(x[i:i+size,:])
    return np.array(data)
size=5
samsung_x=split_data(samsung_x, size)
bit_x=split_data(bit_x, size)
bit_x=bit_x[:samsung_x.shape[0],:]


# y 데이터 추출
samsung_y=samsung_y[5:, :]

# predict 데이터 추출
samsung_x_predict=samsung_x[-1]
bit_x_predict=bit_x[-1]
samsung_x=samsung_x[:-1, :, :]
bit_x=bit_x[:-1, :, :]



samsung_x=samsung_x.astype('float32')
samsung_y=samsung_y.astype('float32')
samsung_x_predict=samsung_x_predict.astype('float32')
bit_x=bit_x.astype('float32')
bit_x_predict=bit_x_predict.astype('float32')


np.save('./data/samsung_x.npy', arr=samsung_x)
np.save('./data/samsung_x_predict.npy', arr=samsung_x_predict)
np.save('./data/samsung_y.npy', arr=samsung_y)
np.save('./data/bit_x.npy', arr=bit_x)
np.save('./data/bit_x_predict.npy', arr=bit_x_predict)


# train, test 분리
samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test=train_test_split(samsung_x, samsung_y, train_size=0.8)
bit_x_train, bit_x_test=train_test_split(bit_x, train_size=0.8)

# print(samsung_x.shape) #(621, 5, 4)
# print(samsung_y.shape) #(621, 1)
# print(bit_x.shape) #(621, 5, 5)

samsung_x_predict=samsung_x_predict.reshape(1,5,4)
bit_x_predict=bit_x_predict.reshape(1,5,5)



######### 2. LSTM 회귀모델
samsung_input1=Input(shape=(5,4))
samsung_layer1=LSTM(100, activation='relu')(samsung_input1)
samsunt_layer1=Dropout(0.2)(samsung_layer1)
samsung_layer2=Dense(500, activation='relu')(samsung_layer1)
samsung_layer3=Dense(3000, activation='relu')(samsung_layer2)
samsunt_layer4=Dropout(0.2)(samsung_layer3)
samsung_layer5=Dense(200, activation='relu')(samsunt_layer4)
samsung_layer6=Dense(20, activation='relu')(samsung_layer5)
samsung_layer6=Dense(10, activation='relu')(samsung_layer6)
samsung_output=Dense(1)(samsung_layer6)



bit_input1=Input(shape=(5,5))
bit_layer1=LSTM(30, activation='relu')(bit_input1)
bit_layer2=Dense(200,activation='relu')(bit_layer1)
bit_layer3=Dense(2000,activation='relu')(bit_layer2)
bit_layer3=Dropout(0.2)(bit_layer3)
bit_layer4=Dense(200,activation='relu')(bit_layer3)
bit_layer5=Dense(20,activation='relu')(bit_layer4)
bit_layer6=Dense(10,activation='relu')(bit_layer5)
bit_layer7=Dense(5,activation='relu')(bit_layer6)
bit_output=Dense(1)(bit_layer7)


merge1=concatenate([samsung_output, bit_output])

output1=Dense(300)(merge1)
output2=Dense(3000)(output1)
output3=Dense(800)(output2)
output4=Dense(30)(output3)
output5=Dense(1)(output4)

model=Model(inputs=[samsung_input1, bit_input1], outputs=output5)

model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es=EarlyStopping(monitor='val_loss',  patience=50, mode='auto')
modelpath='./model/samsung-{epoch:02d}-{val_loss:.4f}.hdf5'
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit([samsung_x_train, bit_x_train], samsung_y_train, epochs=10000, batch_size=1000, validation_split=0.2, callbacks=[es, cp])


#4. 평가, 예측
loss=model.evaluate([samsung_x_test, bit_x_test], samsung_y_test, batch_size=1000)
samsung_y_predict=model.predict([samsung_x_predict, bit_x_predict])

print("loss : ", loss)
print("2020.11.20. 삼성전자 종가 :" , samsung_y_predict)