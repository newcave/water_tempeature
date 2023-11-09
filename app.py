import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 가상 데이터 생성 (임의로 생성)
# 예시 데이터 (날짜, 강우량, 기온, 구름의 양, 수온)
data = {
    '날짜': pd.date_range('20230101', periods=100),
    'rainfall': [i % 10 for i in range(100)],
    '기온': [20 + i % 5 for i in range(100)],
    '구름의양': [30 - i % 6 for i in range(100)],
    '수온': [15 + 0.6 * (i % 8) + 0.3 * (i % 10) for i in range(100)]
}

# 데이터프레임 생성
df = pd.DataFrame(data)

st.title('Linear Regression Model')

# 사이드바에서 학습 및 테스트 섅트 비율 선택
test_sizes = [i / 10 for i in range(1, 10)]
test_size = st.sidebar.selectbox('Select Test Set Size', options=test_sizes)

# Features(특징) 및 Target(목표) 데이터 설정
features = ['rainfall', '기온', '구름의양']
target = '수온'

X = df[features]
y = df[target]

# 학습 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# 선형 회귀 모델 초기화 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
predictions = model.predict(X_test)

# 모델 성능 측정
mse = mean_squared_error(y_test, predictions)
st.write(f"Mean Squared Error: {mse}")

# 그래프 생성
fig, ax = plt.subplots()
ax.scatter(y_test, predictions, label='compare', color='orange')
ax.set_xlabel('test measured')
ax.set_ylabel('model predicted')
ax.set_title('measured vs predicted')

# 선형 추세선 추가
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
ax.plot(y_test, p(y_test), color='red')

# R2 값 표시
r2 = r2_score(y_test, predictions)
ax.text(0.1, 0.9, f'R2 = {r2:.2f}', ha='center', va='center', transform=ax.transAxes, fontsize=10)

st.pyplot(fig)
