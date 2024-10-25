import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def train_and_evaluate(X_train, X_test, y_train, y_test):
    # 모델 생성 및 학습
    model = RandomForestClassifier(n_estimators=10000, random_state=42)
    model.fit(X_train, y_train)

    # 학습 데이터에 대한 성능 평가
    y_train_pred = model.predict(X_train)
    train_report = classification_report(y_train, y_train_pred, target_names=le.classes_)

    # 테스트 데이터에 대한 성능 평가
    y_test_pred = model.predict(X_test)
    test_report = classification_report(y_test, y_test_pred, target_names=le.classes_)

    # 성능 지표 계산
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # 성능 지표 시각화
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    train_scores = [train_accuracy, train_precision, train_recall, train_f1]
    test_scores = [test_accuracy, test_precision, test_recall, test_f1]

    plt.figure(figsize=(12, 6))
    plt.bar(metrics, train_scores, color='skyblue', alpha=0.5, label='Train')
    plt.bar(metrics, test_scores, color='orange', alpha=0.5, label='Test')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.legend()
    plt.show()

    # 결과 출력
    print("Train Classification Report:")
    print(train_report)
    print("Test Classification Report:")
    print(test_report)

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

# 학습 및 평가 함수 호출
train_and_evaluate(X_train, X_test, y_train, y_test)
