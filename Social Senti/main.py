import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from konlpy.tag import Okt


def preprocess_text(text, okt):
    """
    한국어 텍스트 전처리 함수:
    - 구두점 제거
    - 숫자 제거
    - Okt 형태소 분석기를 사용하여 토큰화 후 공백으로 결합
    """
    if pd.isnull(text):  # Null 값 처리
        return ""
    text = re.sub(r'[{}]'.format(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    tokens = okt.morphs(text)
    return " ".join(tokens)


def load_data(file_path):
    """
    TSV 파일을 불러오는 함수.
    파일에는 id, document, label 컬럼이 포함됨.
    """
    df = pd.read_csv(file_path, sep='\t')
    df = df.dropna(subset=['document'])  # Null 값 제거
    return df[['document', 'label']]


def main():
    okt = Okt()

    # 학습 및 테스트 데이터 로드
    train_file = './raw/ratings_train.txt'
    test_file = './raw/ratings_test.txt'

    train_df = load_data(train_file)
    test_df = load_data(test_file)

    print("학습 데이터 수:", len(train_df))
    print("테스트 데이터 수:", len(test_df))

    # 전처리
    train_df['clean_text'] = train_df['document'].apply(lambda x: preprocess_text(x, okt))
    test_df['clean_text'] = test_df['document'].apply(lambda x: preprocess_text(x, okt))

    # TF-IDF 벡터라이저로 텍스트 데이터 수치화
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(train_df['clean_text'])
    X_test_vec = vectorizer.transform(test_df['clean_text'])

    y_train = train_df['label']
    y_test = test_df['label']

    # 랜덤포레스트 분류기로 모델 학습
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_vec, y_train)

    # 테스트 데이터 예측 및 평가
    y_pred = clf.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    # 평가 결과 출력
    print("\n정확도:", accuracy)
    print("\n분류 보고서:")
    print(report)

    # 분류 보고서 시각화
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Mac 환경에서 한글 폰트 설정
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 6))
    sns.heatmap(report_df.iloc[:-3, :-1], annot=True, cmap="Blues", fmt=".2f")
    plt.title("분류 성능 평가 (정밀도, 재현율, F1-Score)")
    plt.yticks(rotation=0)
    plt.show()

    # 사용자 입력 감성 분석 테스트
    print("\n감성 분석 테스트 (문장을 입력하세요. '종료' 입력 시 종료됩니다.)")
    while True:
        user_input = input("입력 문장: ")
        if user_input.strip().lower() == '종료':
            break
        user_input_clean = preprocess_text(user_input, okt)
        user_vec = vectorizer.transform([user_input_clean])
        pred = clf.predict(user_vec)[0]
        sentiment = "긍정" if pred == 1 else "부정"
        print("예측 감성:", sentiment)


if __name__ == "__main__":
    main()