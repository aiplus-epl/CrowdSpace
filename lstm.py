import pandas as pd
import warnings

# 경고 메시지 중 FutureWarning을 무시하기 위한 설정
warnings.simplefilter(action='ignore', category=FutureWarning)

# 파일 경로 설정
file_name = ""
# CSV 파일을 pandas dataframe으로 불러오기 (첫번째 컬럼을 인덱스로 사용)
data = pd.read_csv(file_name, index_col=0)

# 데이터 전처리 시작
# 일본과 미국의 주중, 주말, 공휴일을 구분하여 'day_type' 컬럼에 할당
data['day_type'] = -1  # 일단 모든 값을 -1로 초기화

# 주중일 경우 day_type을 0으로 설정
data.loc[data['day'].isin([1,2,3,4,7,8,9,10,14,15]), 'day_type'] = 0
# 주말일 경우 day_type을 1로 설정
data.loc[data['day'].isin([5, 6, 12, 13]), 'day_type'] = 1
# 공휴일일 경우 day_type을 2로 설정
data.loc[data['day'] == 11, 'day_type'] = 2

# 변환된 데이터 출력
print(data)

import numpy
# 변환된 데이터를 새로운 CSV 파일로 저장
data.to_csv('', index=False)

file_name = ""
data = pd.read_csv(file_name, index_col=0)
print(data)

# 데이터 인덱스 리셋
data = data.reset_index(level=0)
print(data)

# LSTM 모델 학습 시작
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# 데이터 불러오기
file_name2 = ''
data = pd.read_csv(file_name2)

# 데이터 셔플링 (섞기)
shuffled_data = data.sample(frac=1, random_state=42)
print(shuffled_data)

# 'person' 컬럼의 데이터를 2D 배열로 변환
values = data['person'].values.reshape(-1, 1)

# 데이터 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)

# 시퀀스 데이터 생성
look_back = 10  # 과거 10개의 데이터를 사용하여 다음 데이터를 예측
X, y = [], []
for i in range(len(scaled_values) - look_back):
    X.append(scaled_values[i:i+look_back, 0])
    y.append(scaled_values[i+look_back, 0])
X, y = np.array(X), np.array(y)

# 70%는 훈련용, 나머지 30%는 테스트용으로 데이터 분할
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTM 모델 구조 정의
# LSTM(Long Short-Term Memory)은 RNN의 변형으로 긴 시퀀스 데이터의 패턴을 기억하는 데 특화된 구조를 가짐
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))  # LSTM 레이어: 50개의 뉴런, 입력 형태는 (10, 1)
model.add(Dense(1))  # 출력 레이어: 1개의 뉴런 (회귀 예측값 출력)
model.compile(loss='mean_squared_error', optimizer='adam')  # 모델 컴파일. 손실함수는 MSE, 옵티마이저는 Adam

# 모델 학습 시작
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)

# 학습된 모델로 훈련 데이터와 테스트 데이터 예측
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 예측 결과 역정규화
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# 예측 결과 시각화
plt.plot(data.index[:train_size], y_train.flatten(), label='실제 훈련 데이터')
plt.plot(data.index[train_size+look_back:], y_test.flatten(), label='실제 테스트 데이터')
plt.plot(data.index[look_back:train_size+look_back], train_predict.flatten(), label='훈련 데이터 예측')
plt.plot(data.index[train_size+look_back:], test_predict.flatten(), label='테스트 데이터 예측')
plt.legend()
plt.show()

# RMSE 계산을 위한 라이브러리 추가
from sklearn.metrics import mean_squared_error

# 훈련 데이터와 테스트 데이터에 대한 RMSE 계산
train_rmse = np.sqrt(mean_squared_error(y_train[0], train_predict.flatten()))
test_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict.flatten()))

# RMSE 결과 출력
print(f"훈련 데이터 RMSE: {train_rmse:.2f}")
print(f"테스트 데이터 RMSE: {test_rmse:.2f}")
