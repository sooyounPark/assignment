
#BERT 모델 가지치기 실습 (사전학습 가중치 없이 바로 실행)
import torch
from torch.nn.utils import prune
from transformers import BertTokenizer, BertForSequenceClassification

# 사전학습된 BERT 모델 및 토크나이저 로드 (자동 다운로드)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)

# 임의 입력 문장 토크나이즈 및 텐서 변환
text = "이 문장은 BERT 가지치기 실습을 위한 예시입니다."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# 가지치기 전 일부 레이어 가중치 출력
print("가지치기 전:", model.bert.encoder.layer[0].attention.self.key.weight.data[0][:5])

# 가지치기 대상 파라미터 지정 (강의 예제와 동일)
parameters = [
    (model.bert.embeddings.word_embeddings, "weight"),
    (model.bert.encoder.layer[0].attention.self.key, "weight"),
    (model.bert.encoder.layer[1].attention.self.key, "weight"),
    (model.bert.encoder.layer[2].attention.self.key, "weight"),
]

# 전역 L1 Unstructured 가지치기 적용 (20%)
prune.global_unstructured(
    parameters,
    pruning_method=prune.L1Unstructured,
    amount=0.2
)

# 가지치기 후 일부 레이어 가중치 출력
print("가지치기 후:", model.bert.encoder.layer[0].attention.self.key.weight.data[0][:5])

# 추론 예시
with torch.no_grad():
    outputs = model(**inputs)
    print("로짓 출력:", outputs.logits)

