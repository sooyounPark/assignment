import pandas as pd
import csv
from difflib import SequenceMatcher


# CSV 파일에서 FAQ 데이터를 불러오는 함수
def load_faq_data(file_path="faq.csv"):
    """
    주어진 CSV 파일에서 FAQ 데이터를 로드하여 DataFrame으로 반환하는 함수.
    CSV 파일은 'question'과 'answer' 컬럼을 포함해야 합니다.
    """
    try:
        faq_df = pd.read_csv(file_path)
        if faq_df.empty:
            print("[ERROR] FAQ 데이터가 비어있습니다.")
        else:
            print(f"[INFO] {file_path} 파일에서 FAQ 데이터를 성공적으로 로드했습니다.")
        return faq_df
    except Exception as e:
        print(f"[ERROR] CSV 파일 로드 중 오류 발생: {e}")
        return pd.DataFrame()


# 두 문자열의 유사도를 계산하는 함수
def similar(a, b):
    """
    두 문자열의 유사도를 0과 1 사이의 값으로 계산하여 반환합니다.
    """
    a_str = str(a) if a is not None else ''
    b_str = str(b) if b is not None else ''
    return SequenceMatcher(None, a_str.lower(), b_str.lower()).ratio()


# FAQ 매칭 함수 (사용자 질문과 FAQ 데이터의 질문 간 유사도를 계산)
def find_best_match(user_question, faq_df, threshold=0.7):
    """
    사용자 질문과 FAQ 데이터의 질문들 간 유사도를 비교하여 가장 높은 유사도를 가진 항목을 반환합니다.
    유사도가 threshold 미만이면 (매칭 실패) None을 반환합니다.
    """
    best_match = None
    best_score = 0
    for _, row in faq_df.iterrows():
        score = similar(user_question, row['question'])
        if score > best_score:
            best_score = score
            best_match = row
    if best_score < threshold:
        return None, best_score
    return best_match, best_score


# Seq2Seq 기반 응답 생성 (대화 문맥 반영) placeholder 함수
def generate_response_seq2seq_with_context(user_input, conversation_history):
    """
    학습된 Seq2Seq 모델을 이용하여 대화 문맥을 반영한 응답을 생성하는 함수.
    실제 모델 구현 시에는 conversation_history를 전처리하여 모델 입력으로 사용해야 합니다.
    현재는 예시로 placeholder 응답을 반환합니다.
    """
    # 대화 문맥을 단순 연결하여 디버그용으로 출력
    context_text = " ".join([f"{turn['role']}: {turn['text']}" for turn in conversation_history])
    print(f"[DEBUG] 대화 문맥:\n{context_text}")
    # 예시 응답 (실제 구현 시 모델 결과가 여기에 할당됨)
    # 만약 모델이 응답을 생성하지 못하면 빈 문자열("") 혹은 None을 반환하도록 구현
    return "이 부분은 Seq2Seq 모델이 대화 문맥을 반영하여 생성한 응답입니다."


# 새로운 FAQ 항목을 faq.csv에 추가하는 함수
def append_to_faq_csv(question, answer, file_path="faq.csv"):
    """
    새로운 FAQ 항목을 CSV 파일에 추가합니다.
    이미 동일한 질문이 존재하지 않는 경우에만 추가합니다.
    """
    try:
        with open(file_path, "a", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([question, answer])
        print(f"[INFO] 새 FAQ 항목이 faq.csv에 추가되었습니다: {question}")
    except Exception as e:
        print(f"[ERROR] faq.csv 파일에 항목 추가 중 오류 발생: {e}")


# 하이브리드 챗봇 함수 (FAQ 매칭과 Seq2Seq 생성 결합, 다중 턴 대화 관리 및 FAQ 자동 업데이트)
def hybrid_chatbot():
    """
    FAQ 기반 응답과 Seq2Seq 생성 모델을 결합한 하이브리드 챗봇.
    다중 턴 대화를 위해 사용자와 챗봇의 모든 대화 내용을 기록하며,
    매칭되지 않은 새로운 질문을 faq.csv에 자동 반영합니다.

    응답을 대응하지 못하면 FAQ 항목의 답변으로 "null"을,
    추론된 응답이 있다면 해당 응답 앞에 '*'를 붙여 표출하고 추가합니다.
    """
    conversation_history = []  # 대화 이력을 저장할 리스트

    faq_df = load_faq_data()
    if faq_df is None or faq_df.empty:
        print("[ERROR] FAQ 데이터가 없어 챗봇을 실행할 수 없습니다.")
        return

    print("[INFO] 하이브리드 챗봇을 시작합니다. (종료하려면 'exit' 입력)")
    while True:
        user_input = input("사용자 질문: ").strip()
        if user_input.lower() == "exit":
            print("[INFO] 챗봇을 종료합니다.")
            break

        # 사용자 입력을 대화 이력에 저장
        conversation_history.append({"role": "user", "text": user_input})

        # 1단계: FAQ 매칭 시도
        match, score = find_best_match(user_input, faq_df)
        if match is not None:
            response = match['answer']
            print(f"[FAQ 답변] {response} (유사도: {score:.2f})")
        else:
            # 2단계: Seq2Seq 모델을 통한 응답 생성 (대화 문맥 반영)
            inferred_response = generate_response_seq2seq_with_context(user_input, conversation_history)
            # 응답을 대응하지 못한 경우 (예: 빈 문자열 혹은 None) "null"로 처리
            if inferred_response is None or inferred_response.strip() == "":
                response = "null"
                print("[생성 답변] null")
            else:
                response = "*" + inferred_response
                print(f"[생성 답변] {response}")
            # 새 FAQ 항목으로 사용자 질문과 응답을 추가
            append_to_faq_csv(user_input, response)
            # faq_df 갱신하여 이후 대화에 반영
            faq_df = load_faq_data()

        # 챗봇의 응답을 대화 이력에 저장
        conversation_history.append({"role": "bot", "text": response})


if __name__ == "__main__":
    try:
        hybrid_chatbot()
    except Exception as e:
        print(f"[ERROR] 실행 중 오류 발생: {e}")