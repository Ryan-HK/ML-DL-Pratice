
# Simple LSTM
# Q. data 는 0 ~ 99 까지의 연속된 숫자이고, target은 (1 ~ 101) * 2 로 구성
#   입력 data에 대응하는 출력 data를 예측하는 model을 LSTM 으로 작성
#   연속된 5개의 숫자를 보고 다음 숫자를 알아맞추도록 LSTM 을 이용한 Model 작성

#%%
# Step.01 : import
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


#%%
# Step.02 : 데이터 불러오기
#   Training Data는 직접 생성해 줬다.

numbers = [[i] for i in range(105)]
numbers[:10]
print(f">>> number length : {len(numbers)}")

#%%
data = []
target = []

for i in range(5, len(numbers)):
    data.append(numbers[i-5:i])
    target.append(numbers[i][0] * 2)

data[:5]
target[:5]

#%%
# Step.02 : 데이터 변환
npData = np.array(data, dtype="float32")
npTarget = np.array(target, dtype="float32")

# 스케일링
npData = npData / 100.
npTarget = npTarget / 100.

print(f">>> npData.shape : {npData.shape}")
print(f">>> npTarget.shape : {npTarget.shape}")

#%%
# Step.03 : 모델 생성
model = Sequential()
model.add(
    LSTM(16, input_shape=(5, 1))
)

model.add(Dense(1))

model.summary()

#%%
# Step04 : 모델 컴파일
#   딥런닝 모델을 학습할 준비를 하는 단계


# optimizer :
#   -. 손실 함수(Loss Function)을 최소화하기 위해 가중치(Weight)를 조정하는 알고리즘
# Adam(Adaptive Moment Estimation)
#   -. Momentum 과 RMSprop를 결합한 방식
#   -. 학습률을 자동으로 조정하여, 최적의 방향으로 가중치를 업데이트 -> 튜닝이 쉬움
#   -. 불안정한 Gradient 문제 해결
#   단점
#   -. 특정 문제에 대해서 SGD에 비해 일반화 성능이 떨어질 수 있음
#   -. AdamW 가 최근 선호되고 있음

# **
# 일반적인 CNN, NCP -> Adam, AdamW 추천
# LLM (GPT, BERT) -> AdamW
# 시계열 모델 (RNN, LSTM) -> RMSprop

# 손실 함수 (Loss Function)
#   -. 신경망이 학습할 때, 예측값과 실제값의 차이를 측정하는 함수
# 손실 함수는 회귀(Regression) VS 분류(Classification)

# 회귀(Regression)
# MSE (Mean Squared Error : 평균 제곱 오차)
#   -. 이상치에 민감하지만, 미분이 잘 되어 학습 속도가 빠름
# MAE (Mean Absolute Error : 평균 절대 오차)
#   -. 이상치 영향이 적지만, 미분이 0 or 1 이라서 학습속도가 빠르다.

# 분류(Classification) : 카테고리를 예측하는 문제
# Binary Cross Entropy (이진 분류)
#   -. 0 또는 1
# categorical Cross Entropy (다중 분류용)
#   -. 원-핫 인코딩된 다중 클래스 분류
# Sparse Categorical Cross Entropy
#   -. 원-핫 인코딩 없이 정수 레이블 사용 가능

# 결론 : 현재 문제는 회귀 문제니까 MSE 혹은 MAE를 사용한다.
model.compile(optimizer='adam', loss='mae', metrics=['mae'])



#%%
# Step.05 : 모델 훈련
model.fit(npData, npTarget, epochs=500, validation_split=0.2)

#%%

# expected : 80
test_data = [[35], [36], [37], [38], [39]]
x = np.array(test_data, dtype="float32").reshape(1, -1, 1) / 100
print(">>> x.shape : ", x.shape)

model.predict(x) * 100