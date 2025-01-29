#%%
globals().clear()
#%%
# Step.01 : import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

# pip install yfinance
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

#%%
# Step.02 : 데이터 불러오기

# 야후 파이낸스에서 APPLE 데이터를 가져온다.
appl = yf.download('AAPL', start='2015-01-01', end='2022-03-31', progress=False)

#%%
appl.head()
# 가져온 appl의 type은 Pandas의 DataFrame
print(f">>> appl.type : {type(appl)}")

#%%

# 시각화 -> seaborn 라이브러리 사용
# 장점
#   -. Pandas DataFrame과 잘 결합이 된다.
#   -. Matplotlib보다 간결한 코드로 세련된 그래프를 쉽게 생성이 가능
#   -. 통계적 그래프를 쉽게 그릴 수 있음
import seaborn as sns

#%%
x_index = appl.index
print(f">>> x.shape : {x_index.shape}")
# sns lineplot은 1D 배열을 전달이 필요!!
#   -. squeeze() : 2D 배열 -> 1D 배열로 변환
#   -. 배열에서 불필요한 차원(크기가 1인 차원)을 제거하는 메서드 이다.
y_close = appl['Close'].squeeze()
print(f">>> y.shape : {y_close.shape}")

sns.lineplot(x=x_index, y=y_close, data=appl)
plt.title('APPLE stock price')
plt.show()

#%%
# Step.03 : 훈련 데이터 생성
hist = []
target = []

window = 3

close = appl['Close'].squeeze().values

for i in range(len(close) - window):
    x = close[i: i + window]
    y = close[i + window]
    hist.append(x)
    target.append(y)
#%%
print(">>> hist : ", hist[: 10])
print(">>> hist.type : ", type(hist))
print(">>> target : ", target[: 10])
print(">>> target.type : ", type(target))

#%%
print(hist[1][-1] == target[0])

#%%
# np array 로 변환
np_hist = np.array(hist)
print(">>> np_hist.shape : ", np_hist.shape)
# 출력데이터는 현재 1D -> 2D로 차원 업을 해줘야한다.
#   -. 딥러닝 모델은 출력데이터가 샘플 수 x 출력 크기의 형태를 가져야 한다.
np_target = np.array(target).reshape(-1, 1)
print(">>> np_target.shape : ", np_target.shape)

#%%
# Step.04 : 훈련 / 테스트 데이터 구분

# 테스트 데이터는 100일로 할 예정이다.
split = len(hist) - 100

X_train = np_hist[:split]
X_test = np_hist[split:]
y_train = np_target[:split]
y_test = np_target[split:]

print(">>> X_train shape : ", X_train.shape)
print(">>> X_test shape : ", X_test.shape)
print(">>> y_train shape : ", y_train.shape)
print(">>> y_test shape : ", y_test.shape)


#%%
# Step.05 : 정규화 (Normalization)
#   -. 데이터를 일정한 범위로 변환하여, 모델이 학습하기 쉽게 만드는 과정
#   -. 다른 범위의 가진 변수들 간의 불균형 문제를 해결
# Scaling 예정
# MinMaxScaler
#   -. 최소값을 0, 최대값 1 사이로 정규화를 한다.
#   -. 각 feature는 개별적으로 정규화하며, 결과는 [0, 1] 사이에 위치하게 된다.
#   -. feature_range=(a, b)로 결과 범위를 조절 가능 -> 신경망 활성화 함수가 tanh 일 때, 사용된다.
sc1 = MinMaxScaler()

# fit_transform() VS transform()
#   -. fit_transform() : SC 기준으로 학습(fit) 하고 변환
#   -. transform() : 이미 학습한 기준으로 변환
# 테스트 데이터와 훈련 데이터와 다른 기준으로 변환하지 않도록 하기 위해 위와 같이 한다.
X_train_scaled = sc1.fit_transform(X_train)
X_test_scaled = sc1.transform(X_test)

sc2 = MinMaxScaler()
y_train_scaled = sc2.fit_transform(y_train)
y_test_scaled = sc2.transform(y_test)

#(1721, 3)
print(">>> X_train_scaled shape : ", X_train_scaled.shape)
#(100, 3)
print(">>> X_test_scaled shape : ", X_test_scaled.shape)

#%%
# RNN / LSTM 의 모델 입력방식 -> (배치크기, 시퀀스 길이, 특징 수)
X_train_scaled_reshape = X_train_scaled.reshape(-1, window, 1)
X_test_scaled_reshape = X_test_scaled.reshape(-1, window, 1)
print(">>> X_train_scaled_reshape.shape : ", X_train_scaled_reshape.shape)
print(">>> X_test_scaled_reshape.shape : ", X_test_scaled_reshape.shape)

#%%
# Step.06 : 모델 생성

# LSTM Layer
#   -. 시계열 데이터 혹은 연속적인 데이터를 다루는 층 이다.
#   -. 시계열 데이터는 순차적인 의존성을 가지고 있기 때문에, RNN(LSTM)이 적합하다.
# return_sequences=True
#   -. 다층구조에서 모든 타임스텝(시점)의 출력을 다름 LSTM 레이어로 전달할 때 사용한다.
model = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(window, 1), dropout=0.2),
    LSTM(units=32, return_sequences=True, dropout=0.2),
    LSTM(units=16, dropout=0.2),
    Dense(1)
])

model.summary()


#%%
# Step.07 : 모델 학습

# Compile
model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train_scaled_reshape, y_train_scaled, epochs=30, batch_size=16)

#%%
# Step.08 : 시각화
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('mse')
plt.show()

#%%
# Step.09 : 평가
pred = model.predict(X_test_scaled_reshape)

plt.figure(figsize=(12, 6))
plt.plot(np.concatenate((y_train_scaled.flatten(), y_test_scaled.flatten()), axis=0))
plt.plot(np.concatenate((y_train_scaled.flatten(), pred.flatten()), axis=0))
plt.show()

#%%
# 정규화 해제
plt.figure(figsize=(12, 6))
pred_original = sc2.inverse_transform(pred)
y_train_scaled_original = sc2.inverse_transform(y_train_scaled)
y_test_original = sc2.inverse_transform(y_test_scaled)

plt.plot(np.concatenate((y_train_scaled_original.flatten(), y_test_original.flatten()), axis=0))
plt.plot(np.concatenate((y_train_scaled_original.flatten(), pred_original.flatten()), axis=0))
plt.show()