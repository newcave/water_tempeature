import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 가상 데이터 생성
data = {
    '날짜': pd.date_range('20230101', periods=100),
    'rainfall': [i % 10 for i in range(100)],
    '기온': [20 + i % 5 for i in range(100)],
    '구름의양': [30 - i % 6 for i in range(100)],
    '수온': [15 + 0.6 * (i % 8) + 0.3 * (i % 10) for i in range(100)]
}

# 데이터프레임 생성
df = pd.DataFrame(data)

# Features(특징) 및 Target(목표) 데이터 설정
features = ['rainfall', '기온', '구름의양']
target = '수온'

X = df[features]
y = df[target]

# 학습 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 초기화 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
predictions = model.predict(X_test)

# 모델 성능 측정
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Streamlit 앱 구성
st.title('Linear Regression Model Results')
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R2 Score: {r2:.2f}")

# 테스트 데이터에서의 실제 값과 모델 예측 값 비교 그래프
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(y_test, predictions, label='compare', color='orange')
ax.set_xlabel('test measured')
ax.set_ylabel('model predicted')
ax.set_title('measured vs predicted')

# 선형 추세선 추가
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
ax.plot(y_test, p(y_test), color='red')

st.pyplot(fig)
