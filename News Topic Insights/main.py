import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import LatentDirichletAllocation

# 1. 데이터셋 불러오기
# 'headers', 'footers', 'quotes'를 제거하여 텍스트 노이즈를 줄입니다.
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
print("불러온 데이터 개수:", len(newsgroups.data))
print("분류 라벨:", newsgroups.target_names)

# 2. 텍스트 전처리 및 TF-IDF 벡터화
# stop_words 제거 및 max_df 설정을 통해 자주 등장하는 단어의 영향력을 줄입니다.
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

# 3. 뉴스 기사 주제 분류: 랜덤포레스트 분류기 적용
# 데이터를 학습용과 테스트용으로 분리합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤포레스트 분류기를 학습합니다.
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 테스트 데이터로 예측을 수행하고 성능 평가를 진행합니다.
y_pred = clf.predict(X_test)
print("\n--- 뉴스 기사 주제 분류 성능 평가 ---")
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))

# 4. 토픽 모델링: LDA를 활용하여 뉴스 기사 내 잠재적 토픽 추출
# 전체 데이터를 대상으로 LDA 모델을 학습합니다.
n_topics = 10  # 토픽의 수는 필요에 따라 조정
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)

# 각 토픽별로 상위 단어들을 출력하는 함수 정의
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic {topic_idx + 1}:")
        # 상위 단어들을 추출 (단어 index 기준 내림차순 정렬)
        top_features_ind = topic.argsort()[:-no_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        print(" ".join(top_features))

print("\n--- 토픽 모델링 결과 ---")
feature_names = vectorizer.get_feature_names_out()
display_topics(lda, feature_names, no_top_words=10)