# Simple Linear Regression (선형회귀)
#   -. 독립변수(특징, X) 와 종속변수(타겟, y) 간의 관계를 직선 방정식으로 모델링하는 기법

#%%
#-----------------------------------------------------------------
# Step.01 : Import libraries
#-----------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


#%%
#-----------------------------------------------------------------
# Step.02 : Datasets 불러오기
#-----------------------------------------------------------------
dia = datasets.load_diabetes()
print(dia.feature_names)
print(dia.data.shape)
print(dia.target.shape)

#%%
# DataFrame으로 만들기
#   -. Pandas의 DataFrame은 테이블 형태의 데이터를 다루는 Python 객체이다.
#   -. 행렬 형태의 데이터를 표 형식으로 변환하는 작업
df = pd.DataFrame(data=dia.data, columns=dia.feature_names)
print(df.head())
print(df.shape)

#%%
#-----------------------------------------------------------------
# Step.03 : 머신러닝 작업에 적합한 형태로 데이터 가공
#-----------------------------------------------------------------
# df['bmi'] -> Pandas의 Series 객체
# .values -> Numpy 배열로 변환 (1차원 배열 : 벡터)
# .reshape -> 차원변경 : 2차원 배열로 변환
#   -. 머신러닝 라이브러리의 입력데이터는 2차원 배열 형식이다.
#   -. (n_samples, n_features)
dia_X = df['bmi'].dia_X.reshape(-1, 1)
print(">>> shape : ", dia_X.shape)
print(">>> type : ", type(dia_X))


#%%
#-----------------------------------------------------------------
# Step.04 : 데이터셋 분리 작업
#-----------------------------------------------------------------
# Train, Test Set을 Split
#   -. 추후에는 scikit-learn의 train_test_split 함수를 사용해서 분리가 쉽게 가능하다!
dia_X_train = dia_X[:-20]
dia_X_test = dia_X[-20:]
print(">>> shape : ", dia_X_train.shape)
print(">>> shape : ", dia_X_test.shape)
print(">>> type : ", type(dia_X_train))

# 정답데이터
dia_y_train = dia.target[:-20]
dia_y_test = dia.target[-20:]
print(">>> shape : ", dia_y_train.shape)
print(">>> shape : ", dia_y_test.shape)


#%%
#-----------------------------------------------------------------
# Step.05 : 학습
#-----------------------------------------------------------------
regr = linear_model.LinearRegression()

# ML의 입력데이터는 2차원의 numpy 배열이 필요하다!!
regr.fit(dia_X_train, dia_y_train)

#%%
#-----------------------------------------------------------------
# Step.06 : 예측
#-----------------------------------------------------------------
# 테스트 데이터 제공
y_pred = regr.predict(dia_X_test)


#%%
#-----------------------------------------------------------------
# Step.07 : 시각화
#-----------------------------------------------------------------
plt.scatter(dia_X_test, dia_y_test, label='True Value')
plt.plot(dia_X_test, y_pred, color='r', label='Predicted Value')
plt.xlabel('bmi')
plt.ylabel('Progress')
plt.legend()
plt.show()


#%%
# R2 Score (결정계수)
#   특징
#   -. 회귀 모델의 성능을 평가하는 지표로 모델이 타겟 변수를 얼마나 잘 설명하였는지를 나타낸다.
#   -. 0에서 1사이의 값을 가진다.
#   장점
#   -. 데이터의 분산을 기준으로 모델의 성능을 평가할 수 있음
#   단점
#   -. Overfitting(과적합) 문제 발생
#   -. 데이터와 모델간의 선형관계를 가정하므로, 비선형 모델에서는 비적합할 수 있음
r2 = r2_score(dia_y_test, y_pred)
print(">>> r2_score : ", r2)

mean_squared_error = mean_squared_error(dia_y_test, y_pred)
print(">>> mean_squared_error : ", mean_squared_error)

#%%
#-----------------------------------------------------------------
# 추가 응용 : 다변수로 계산하기
#-----------------------------------------------------------------
# 다변수로 계산하기
# dia_X = df[['bmi', 'bp']].values

dia_X = df.values
print(">>> shape : ", dia_X.shape)

# Train, Test Set을 Split
#   -. 추후에는 scikit-learn의 train_test_split 함수를 사용해서 분리가 쉽게 가능하다!
dia_X_train = dia_X[:-20]
dia_X_test = dia_X[-20:]
print(">>> shape : ", dia_X_train.shape)
print(">>> shape : ", dia_X_test.shape)

# 정답데이터
dia_y_train = dia.target[:-20]
dia_y_test = dia.target[-20:]
print(">>> shape : ", dia_y_train.shape)
print(">>> shape : ", dia_y_test.shape)

# 모델 훈련
regr = linear_model.LinearRegression()
regr.fit(dia_X_train, dia_y_train)

# 예측
y_pred = regr.predict(dia_X_test)

# R2 Score
r2 = r2_score(dia_y_test, y_pred)
print(">>> r2_score : ", r2)

mse = mean_squared_error(dia_y_test, y_pred)
print(">>> mean_squared_error : ", mse)

#%%
globals().clear()