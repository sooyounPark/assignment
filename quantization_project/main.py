from gensim.models import Word2Vec
from konlpy.tag import Okt
#
# 간단한 한글 문장 코퍼스 준비
corpus = [
    "자연어 처리 기술은 매우 빠르게 발전하고 있습니다.",
    "딥러닝 모델을 사용한 언어 모델이 다양한 분야에서 활용됩니다.",
    "텍스트 데이터의 임베딩은 NLP 작업의 핵심 요소입니다."
]

# 형태소 분석을 통한 토큰화
okt = Okt()
tokenized_corpus = [okt.morphs(sent) for sent in corpus]

# Word2Vec 모델 학습
model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=50,    # 임베딩 차원
    window=2,          # 윈도우 크기
    min_count=1,       # 최소 등장 빈도
    workers=1,         # 멀티프로세싱 비활성화(환경 호환)
    sg=1               # skip-gram 방식
)

# 단어 임베딩 확인
print("['자연어'] 임베딩 벡터:", model.wv['자연어'])

# 단어 유사도 확인
print("['자연어', '모델'] 유사도:", model.wv.similarity('자연어', '모델'))

# 가장 유사한 단어 3개 출력
print("['자연어']와 유사한 단어:", model.wv.most_similar('자연어', topn=3))
