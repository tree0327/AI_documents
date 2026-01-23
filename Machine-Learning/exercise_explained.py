import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ==========================================
# 1교시: 데이터 만들기 (재료 준비)
# ==========================================
print("=== 1교시: 데이터 만들기 ===")

# 컴퓨터가 매번 똑같은 랜덤 숫자를 만들도록 설정 (실험 결과를 똑같이 만들기 위해)
np.random.seed(0)

# X는 '공부한 시간'이라고 생각해봅시다.
# 0시간부터 2시간 사이의 랜덤한 시간들을 100명분 만듭니다.
# rand(100, 1)은 0~1 사이의 숫자를 100개 만드는데, 여기에 2를 곱했으니 0~2 사이가 됩니다.
X = 2 * np.random.rand(100, 1)

# y는 '시험 점수'라고 생각해봅시다.
# 점수 = 6 + 4 * (공부한 시간) + (약간의 운)
# 원래는 4시간 공부하면 4점 올라야 하는데, 사람마다 컨디션(노이즈)이 달라서 조금씩 차이가 납니다.
# randn(100, 1)은 정규분포(평균 0)를 따르는 무작위 잡음입니다.
y = 6 + 4 * X + np.random.randn(100, 1)

print(f"학생 1의 공부 시간: {X[0][0]:.2f}시간, 점수: {y[0][0]:.2f}점")
print(f"학생 2의 공부 시간: {X[1][0]:.2f}시간, 점수: {y[1][0]:.2f}점")
print("... 총 100명의 데이터가 준비되었습니다.\n")


# ==========================================
# 2교시: Scikit-learn으로 쉽게 예측하기 (자동모드)
# ==========================================
print("=== 2교시: AI 라이브러리(Scikit-learn)로 예측하기 ===")

# 데이터를 두 편으로 나눕니다.
# 1. 훈련용 데이터(Train Set): AI가 공부할 문제집 (80%)
# 2. 테스트용 데이터(Test Set): AI가 시험 볼 시험지 (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# 1. 모델 준비 (빈 뇌를 가진 A.I 로봇 데려오기)
# LinearRegression은 직선을 그어서 정답을 맞추는 가장 기본적인 AI 모델입니다.
model = LinearRegression()

# 2. 모델 학습 (공부 시키기)
# "이 시간(X_train) 공부하면 이 점수(y_train)가 나와. 잘 봐둬!"
model.fit(X_train, y_train)

# 3. 모델 예측 (시험 보기)
# "자, 이제 이 시간(X_test) 공부한 학생은 몇 점 맞을 것 같아?"
y_pred = model.predict(X_test)

# 4. 채점하기 (얼마나 잘 맞췄나?)
# MSE(평균 제곱 오차): 틀린 점수의 제곱의 평균. 0에 가까울수록 좋습니다. (작을수록 좋음)
mse = mean_squared_error(y_test, y_pred)
# R2 Score(결정 계수): 100점 만점에 몇 점짜리 모델인지. 1에 가까울수록 좋습니다. (클수록 좋음)
r2 = r2_score(y_test, y_pred)

print(f"AI 모델의 성적표:")
print(f" - 오차(MSE): {mse:.4f} (작을수록 좋아요)")
print(f" - 정확도(R2): {r2:.4f} (1.0에 가까울수록 완벽해요)")
print(f" - AI가 찾은 공식: 점수 = {model.coef_[0][0]:.2f} * 시간 + {model.intercept_[0]:.2f}")
print(f" - (원래 우리가 만든 공식: 점수 = 4 * 시간 + 6)")
print(f" -> 꽤 비슷하게 맞췄죠?\n")


# ==========================================
# 3교시: 경사하강법 직접 해보기 (수동모드 - 원리 이해)
# ==========================================
print("=== 3교시: 경사하강법(Gradient Descent)으로 직접 정답 찾아가기 ===")
print("설명: 경사하강법은 산에서 눈을 가리고 가장 낮은 곳(오차가 가장 적은 곳)으로 조금씩 내려가는 방법입니다.")

# 비용 함수 (틀린 정도 계산기)
# 우리가 그은 선이 정답이랑 얼마나 차이가 나는지 계산합니다.
def get_cost(y, y_pred):
    N = len(y)
    # (실제값 - 예측값)을 제곱해서 다 더하고 평균을 냅니다.
    cost = np.sum(np.square(y - y_pred)) / N
    return cost

# 가중치 업데이트 함수 (다음 발자국 어디로 딛을지 계산)
# w1: 기울기 (공부 시간이 점수에 미치는 영향)
# w0: 절편 (공부 안 해도 기본으로 나오는 점수)
# learning_rate: 보폭 (한 번에 얼마나 수정할지)
def get_weight_update(w1, w0, X, y, learning_rate=0.01):
    N = len(y)
    
    # 현재 w1, w0로 예측해봅니다.
    y_pred = np.dot(X, w1.T) + w0
    
    # 실제값과 얼마나 차이 나는지 봅니다.
    diff = y - y_pred
    
    # 미분(기울기)을 이용해서 오차를 줄이는 방향을 찾습니다. (이 공식은 수학적으로 유도된 것입니다)
    # w1을 얼마나 바꿀지
    w1_update = -(2 / N) * learning_rate * (np.dot(X.T, diff))
    # w0를 얼마나 바꿀지
    w0_factors = np.ones((N, 1))
    w0_update = -(2 / N) * learning_rate * (np.dot(w0_factors.T, diff))
    
    return w1_update, w0_update

# 경사하강법 시작!
def gradient_descent_steps(X, y, iters=1000):
    # 처음에는 아무것도 모르니까 0부터 시작합니다.
    w0 = np.zeros((1, 1))
    w1 = np.zeros((1, 1))
    
    # iters 만큼 반복해서(공부해서) 조금씩 정답에 가까워집니다.
    for ind in range(iters):
        # 이번에 얼마나 수정할지 계산
        w1_update, w0_update = get_weight_update(w1, w0, X, y, learning_rate=0.01)
        
        # 수정사항 반영 (원래 값에서 업데이트 값을 뺍니다)
        w1 = w1 - w1_update
        w0 = w0 - w0_update
        
        # 100번마다 얼마나 잘하고 있나 로그 출력
        if ind % 100 == 0:
             # 현재 예측값
            y_pred = w1[0, 0] * X + w0
            # 현재 오차
            cost = get_cost(y, y_pred)
            print(f"[{ind}번 반복] 현재 오차: {cost:.4f} / 기울기(w1): {w1[0,0]:.2f} / 절편(w0): {w0[0,0]:.2f}")
            
    return w1, w0

# 실행해봅시다 (1000번 반복 학습)
print("-> 학습 시작! (오차가 줄어드는 것을 확인하세요)")
w1_final, w0_final = gradient_descent_steps(X, y, iters=1000)

print(f"\n최종 결과:")
print(f" - 경사하강법으로 찾은 기울기(w1): {w1_final[0, 0]:.4f} (정답인 4에 가깝나요?)")
print(f" - 경사하강법으로 찾은 절편(w0): {w0_final[0, 0]:.4f} (정답인 6에 가깝나요?)")


# ==========================================
# 4교시: 눈으로 확인하기 (시각화)
# ==========================================
print("\n=== 4교시: 그래프 그려보기 ===")
# 1. 원래 데이터 점 찍기
plt.scatter(X, y, label='Real Data (Students)')

# 2. 우리가 찾은 AI 모델(붉은 선) 그리기
# X축 값들에 대해 우리가 찾은 w1, w0로 예측값을 계산해서 선을 긋습니다.
y_pred_line = w1_final[0, 0] * X + w0_final
plt.plot(X, y_pred_line, color='red', label='AI Model')

plt.xlabel('Study Hours (X)')
plt.ylabel('Score (y)')
plt.title('Study Hours vs Score')
plt.legend()
plt.savefig('result_graph.png') # 그래프를 파일로 저장
print("그래프가 'result_graph.png' 파일로 저장되었습니다.")
# plt.show() # 화면에 띄우기 (터미널 환경에서는 생략 가능)
