from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

# 예시 데이터셋: 실제로는 더 많은 데이터와 올바른 정규화된 주소가 필요합니다.
data = [
    {"raw_address": "장흥소방서", "normalized_address": "전라남도 장흥군 장흥읍 중앙로 123"},
    {"raw_address": "전남 영광군 중앙로 123-1, 석가아파트 103동 102호", "normalized_address": "전라남도 영광군 중앙로 123-1 석가아파트 103동 102호"},
    # 추가 데이터...
]

# 데이터셋 생성
dataset = Dataset.from_list(data)

# T5 토크나이저와 모델 불러오기 (필요시 한국어 특화 모델로 교체 가능)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# 전처리 함수: 입력 텍스트와 정답 텍스트를 토크나이즈합니다.
def preprocess_function(examples):
    # 입력: "normalize address: <raw_address>" 형태로 구성
    inputs = ["normalize address: " + addr for addr in examples["raw_address"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    # 정답 토큰화
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["normalized_address"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 데이터셋 매핑
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./t5_address_finetuned",
    num_train_epochs=3,                    # 에폭 수 (데이터셋 크기에 따라 조정)
    per_device_train_batch_size=8,         # 배치 사이즈 (메모리 상황에 따라 조정)
    save_steps=500,                        # 모델 저장 주기
    logging_steps=100,                     # 로깅 주기
    evaluation_strategy="no",              # 평가 전략 (예제에서는 생략)
    save_total_limit=2,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 모델 파인튜닝 시작
trainer.train()