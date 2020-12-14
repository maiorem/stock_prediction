import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

######### 1. 데이터
big_x=np.load('./data/monday/lstm_big_x.npy')
big_x_predict=np.load('./data/monday/lstm_big_x_predict.npy')
big_y=np.load('./data/monday/lstm_big_y.npy')


# train, test 분리
big_x_train, big_x_test, big_y_train, big_y_test=train_test_split(big_x, big_y, train_size=0.8)


######### 2. LSTM 회귀모델
model = load_model('./model/samsung-noensemble-lstm-270-1255467.1250.hdf5')


#4. 평가, 예측
loss=model.evaluate(big_x_test, big_y_test, batch_size=100)
samsung_y_predict=model.predict(big_x_predict)

print("loss : ", loss)
print("2020.11.23. 월요일 삼성전자 시가 :" , samsung_y_predict)
