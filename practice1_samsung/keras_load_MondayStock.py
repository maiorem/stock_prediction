import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

######### 1. 데이터
samsung_x=np.load('./data/monday/samsung_x.npy')
samsung_x_predict=np.load('./data/monday/samsung_x_predict.npy')
samsung_y=np.load('./data/monday/samsung_y.npy')
bit_x=np.load('./data/monday/bit_x.npy')
bit_x_predict=np.load('./data/monday/bit_x_predict.npy')
gold_x=np.load('./data/monday/gold_x.npy')
gold_x_predict=np.load('./data/monday/gold_x_predict.npy')
kosdak_x=np.load('./data/monday/kosdak_x.npy')
kosdak_x_predict=np.load('./data/monday/kosdak_x_predict.npy')

# train, test 분리
samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test=train_test_split(samsung_x, samsung_y, train_size=0.8)
bit_x_train, bit_x_test, gold_x_train, gold_x_test, kosdak_x_train, kosdak_x_test=train_test_split(bit_x, gold_x, kosdak_x, train_size=0.8)


samsung_x_predict=samsung_x_predict.reshape(1,5,4)
bit_x_predict=bit_x_predict.reshape(1,5,5)
gold_x_predict=gold_x_predict.reshape(1,5,6)
kosdak_x_predict=kosdak_x_predict.reshape(1,5,3)


######### 2. LSTM 회귀모델
model = load_model('./model/samsung-monday-210-1155323.5000.hdf5')

#4. 평가, 예측
loss=model.evaluate([samsung_x_test, bit_x_test, gold_x_test, kosdak_x_test], samsung_y_test, batch_size=100)
samsung_y_predict=model.predict([samsung_x_predict, bit_x_predict, gold_x_predict, kosdak_x_predict])

print("loss : ", loss)
print("2020.11.23. 월요일 삼성전자 시가 :" , samsung_y_predict)