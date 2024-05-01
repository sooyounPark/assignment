import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 데이터 로드
data = pd.read_csv('ptbl_history.csv')

# 데이터 전처리
data = data[['HST_BFR_ORG_NM', 'HST_ORG_NM']].dropna()

# 레이블 인코딩
le = LabelEncoder()
data['HST_BFR_ORG_NM_encoded'] = le.fit_transform(data['HST_BFR_ORG_NM'])
data['HST_ORG_NM_encoded'] = le.fit_transform(data['HST_ORG_NM'])

# 특성과 타겟 분리
X = data[['HST_BFR_ORG_NM_encoded']]
y = data['HST_ORG_NM_encoded']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성 및 학습
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# 예제 근무지 선택 및 인코딩
example_location = input("원하는 근무지를 입력해주세요: ")
if example_location in le.classes_:
    example_encoded = le.transform([example_location])[0]

    # 예측 데이터 프레임 생성, 컬럼 이름 명시적으로 지정
    example_df = pd.DataFrame([example_encoded], columns=['HST_BFR_ORG_NM_encoded'])

    # 예측 확률 계산
    probabilities = model.predict_proba(example_df)[0]

    # 확률에 따라 근무지 순위 생성
    locations = le.inverse_transform(np.argsort(probabilities)[::-1])
    probabilities_sorted = np.sort(probabilities)[::-1]

    # 결과 시각화
    plt.rc('font', family='AppleGothic')
    plt.figure(figsize=(20, 8))
    plt.barh(locations[:10], probabilities_sorted[:10], color='skyblue')
    plt.xlabel('Probability')
    plt.ylabel('Work Location')
    plt.title(f"{example_location}을 선택한 유저의 예상 선호 근무지 10개")
    plt.gca().invert_yaxis()
    plt.show()
else:
    print(f"{example_location} is not a recognized location in the dataset.")
