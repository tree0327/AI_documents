# ==============================================================================
# 🚢 타이타닉 생존자 예측하기: 인공지능 탐정 놀이 (완전판) 🕵️‍♀️🕵️‍♂️
# ==============================================================================

# 안녕하세요! 아까보다 더 자세하게 탐정 놀이를 해볼 거예요.
# 수첩의 작은 낙서 하나하나 놓치지 않고 꼼꼼하게 살펴볼게요!

# ------------------------------------------------------------------------------
# 1단계: 도구 챙기기 & 데이터 불러오기 📚
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 엑셀 파일(csv)을 읽어서 'titanic_df'라는 탐정 수첩에 옮겨 적어요.
titanic_df = pd.read_csv("./data/titanic_train.csv")

print("--- 1. 탐정 수첩의 첫 5줄 ---")
print(titanic_df.head())

# ------------------------------------------------------------------------------
# 2단계: 수첩 훑어보기 (정보 확인) 👀
# ------------------------------------------------------------------------------
# 데이터가 몇 개인지, 빈칸은 없는지 확인해요.
print("\n--- 2. 수첩 건강 검진 (정보 확인) ---")
titanic_df.info()

# 빈칸(Null)이 몇 개인지 세어볼까요?
print("\n--- 3. 비어있는 칸 개수 세어보기 ---")
print(titanic_df.isnull().sum())

# ------------------------------------------------------------------------------
# 3단계: 빈칸 채우기 (결측치 처리) ✏️
# ------------------------------------------------------------------------------
# 나이(Age)는 평균 나이로, 나머지는 'N'으로 채워줄게요.
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
titanic_df['Cabin'].fillna('N', inplace=True)
titanic_df['Embarked'].fillna('N', inplace=True)

print("\n--- 4. 빈칸 채우기 완료! 확인해볼까요? ---")
print(titanic_df.isnull().sum())

# ------------------------------------------------------------------------------
# 4단계: 데이터 꼼꼼히 뜯어보기 (값 분포 확인) 🔍 [추가된 내용!]
# ------------------------------------------------------------------------------
# 성별, 방 번호, 탄 항구에 어떤 값들이 있는지 세어봐요.
# 남자/여자는 몇 명인지, 방 번호는 어떤 게 많은지 보는 거예요.

print("\n--- 5. 성별(Sex) 분포 확인 ---")
print(titanic_df['Sex'].value_counts())

print("\n--- 6. 방 번호(Cabin) 분포 확인 ---")
print(titanic_df['Cabin'].value_counts())

print("\n--- 7. 탄 항구(Embarked) 분포 확인 ---")
print(titanic_df['Embarked'].value_counts())

# 방 번호(Cabin)가 'C85', 'C123' 처럼 너무 복잡해요.
# 맨 앞 글자(C)가 중요하니까 앞 글자만 남길게요.
titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
print("\n--- 8. 방 번호 앞 글자만 남기기 ---")
print(titanic_df['Cabin'].head())

# ------------------------------------------------------------------------------
# 5단계: 누가 더 많이 살았을까? (그룹별 생존자) 📊 [추가된 내용!]
# ------------------------------------------------------------------------------
# 성별에 따라 살았는지(Survived)를 묶어서(groupby) 세어볼까요?
# 남자(male)와 여자(female) 중 누가 더 많이 1(생존)이 되었을까요?

print("\n--- 9. 성별 별 생존자 수 세어보기 ---")
print(titanic_df.groupby(['Sex','Survived'])['Survived'].count())

# ------------------------------------------------------------------------------
# 6단계: 그림으로 그려보기 (시각화) 🎨
# ------------------------------------------------------------------------------
# 숫자로만 보면 머리 아프니까 그림으로 그려봐요.

print("\n--- 10. 그래프 그리기 (창이 뜨면 닫아주세요) ---")

# 1. 성별에 따른 생존 확률
plt.figure()
sns.barplot(x='Sex', y='Survived', data=titanic_df)
plt.title("Sex vs Survived")
plt.show()

# 2. 좌석 등급(Pclass)과 성별에 따른 생존 확률
# 부자(1등석)와 서민(3등석)의 차이를 봐요.
plt.figure()
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)
plt.title("Pclass vs Survived")
plt.show()

# 3. 나이(Age)별 생존 확률
# 나이를 구분하는 함수를 만들어서 그래프를 그려요.
def get_category(age):
    cat = ''
    if age <= -1: cat = 'Unknown'
    elif age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else : cat = 'Elderly'
    return cat

plt.figure(figsize=(10,6))
group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Elderly']
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))
sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=titanic_df, order=group_names)
plt.title("Age vs Survived")
plt.show()

# 다 쓴 임시 정보는 지워요.
titanic_df.drop('Age_cat', axis=1, inplace=True)

# ------------------------------------------------------------------------------
# 7단계: 척척박사 함수 만들기 (전처리 함수 모음) 🤖 [중요!]
# ------------------------------------------------------------------------------
# 지금까지 했던 정리 작업들을 언제든지 다시 할 수 있게 로봇(함수)으로 만들어둘게요.

from sklearn.preprocessing import LabelEncoder

# 1. 빈칸 채워주는 로봇
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)
    return df

# 2. 필요 없는 정보 버리는 로봇
def drop_features(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df

# 3. 글자를 숫자로 바꿔주는 로봇 (인코딩)
def label_encode(df):
    df['Cabin'] = df['Cabin'].str[:1] # 방 번호 앞글자만
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

# 4. 위 3단계를 한 번에 해주는 대장 로봇
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = label_encode(df)
    return df

# ------------------------------------------------------------------------------
# 8단계: 시험 준비하기 (데이터 나누기) 📝
# ------------------------------------------------------------------------------
# 다시 데이터를 처음부터 불러와서 대장 로봇에게 맡길게요.
titanic_df = pd.read_csv("./data/titanic_train.csv")
y_titanic_df = titanic_df['Survived'] # 정답 (생존 여부)
X_titanic_df = titanic_df.drop('Survived', axis=1) # 문제 (나머지 정보)

# 대장 로봇 출동! 데이터를 숫자로 예쁘게 정리해주세요.
X_titanic_df = transform_features(X_titanic_df)

# 공부용(Train)과 시험용(Test)으로 나눠요 (8:2 비율)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, \
                                                    test_size=0.2, random_state=11)

# 데이터가 잘 나눠졌는지 개수를 확인해봐요. [추가된 내용!]
print("\n--- 11. 데이터 나누기 확인 ---")
print(f"공부할 문제 개수: {len(X_train)}")
print(f"시험볼 문제 개수: {len(X_test)}")
print(f"공부할 정답 개수: {len(y_train)}")
print(f"시험볼 정답 개수: {len(y_test)}")

# ------------------------------------------------------------------------------
# 9단계: 로봇 학생들 공부시키기 (모델 학습) 🏫
# ------------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 3명의 학생 입장!
dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression(solver='liblinear')

print("\n--- 12. 로봇 학생 시험 점수 발표 ---")

# 1번: 결정 트리 학생
dt_clf.fit(X_train, y_train) # 공부하기
dt_pred = dt_clf.predict(X_test) # 시험보기
print(f'DecisionTreeClassifier(결정 트리) 정확도: {accuracy_score(y_test, dt_pred):.4f}')

# 2번: 랜덤 포레스트 학생
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
print(f'RandomForestClassifier(랜덤 포레스트) 정확도: {accuracy_score(y_test, rf_pred):.4f}')

# 3번: 로지스틱 회귀 학생
lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)
print(f'LogisticRegression(로지스틱 회귀) 정확도: {accuracy_score(y_test, lr_pred):.4f}')

# ------------------------------------------------------------------------------
# 10단계: 모의고사 5번 보기 (K-Fold 교차 검증) 🔄
# ------------------------------------------------------------------------------
from sklearn.model_selection import KFold

def exec_kfold(clf, folds=5):
    kfold = KFold(n_splits=folds)
    scores = []
    
    for iter_count, (train_index, test_index) in enumerate(kfold.split(X_titanic_df)):
        X_train, X_test = X_titanic_df.values[train_index], X_titanic_df.values[test_index]
        y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]
        
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        scores.append(accuracy)
        print(f"교차 검증 {iter_count}의 정확도: {accuracy:.4f}")
    
    print(f"## 평균 정확도: {np.mean(scores):.4f}")

print("\n--- 13. [결정 트리] 모의고사(K-Fold) 결과 ---")
exec_kfold(dt_clf, folds=5)

# ------------------------------------------------------------------------------
# 11단계: 자동 모의고사 (cross_val_score) ⏩
# ------------------------------------------------------------------------------
from sklearn.model_selection import cross_val_score

print("\n--- 14. [결정 트리] 자동 모의고사(cross_val_score) 결과 ---")
scores = cross_val_score(dt_clf, X_titanic_df, y_titanic_df, cv=5)
for iter_count, accuracy in enumerate(scores):
    print(f"교차 검증 {iter_count}의 정확도: {accuracy:.4f}")

print(f"## 평균 정확도: {np.mean(scores):.4f}")

# ------------------------------------------------------------------------------
# 12단계: 최고의 아이템 찾기 (GridSearchCV) 💎
# ------------------------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

parameters = {
    'max_depth': [2, 3, 5, 10],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 5, 8]
}

# 5번씩 시험 보면서(cv=5) 제일 좋은 설정을 찾아라!
grid_dclf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=5, refit=True)
grid_dclf.fit(X_train, y_train)

print("\n--- 15. 최고의 아이템 찾기 결과 ---")
print('GridSearchCV 최적 하이퍼 파라미터:', grid_dclf.best_params_)
print(f'GridSearchCV 최고 정확도: {grid_dclf.best_score_:.4f}')

# 최적의 설정으로 마지막 시험을 봐볼까요?
best_dclf = grid_dclf.best_estimator_
dpredictions = best_dclf.predict(X_test)
accuracy = accuracy_score(y_test, dpredictions)
print(f'최종 업그레이드된 DecisionTreeClassifier 정확도: {accuracy:.4f}')

# ==============================================================================
# 미션 성공! 🎉
# 아까 놓쳤던 데이터 확인(value_counts)과 그룹별 비교(groupby)까지 
# 모두 포함해서 완벽하게 분석했어요. 수고했어요! 👍
# ==============================================================================
