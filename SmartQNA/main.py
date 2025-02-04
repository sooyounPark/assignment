from difflib import SequenceMatcher
from loadDataQna import load_data_from_csv


def similar(a, b):
    """
    두 문자열의 유사도를 계산하여 0과 1 사이의 값으로 반환하는 함수.
    문자열이 아닌 경우, 문자열로 변환하여 비교합니다.
    """
    # None 체크 및 문자열 변환
    a_str = str(a) if a is not None else ''
    b_str = str(b) if b is not None else ''
    return SequenceMatcher(None, a_str.lower(), b_str.lower()).ratio()

def get_faq_data(file_path="faq.csv"):
    """
    CSV 파일에서 FAQ 데이터를 로드하는 함수.
    CSV 파일은 'question'과 'answer' 두 개의 컬럼을 포함해야 함.
    """
    faq_df = load_data_from_csv(file_path)
    if faq_df.empty:
        print("[ERROR] FAQ 데이터가 존재하지 않습니다.")
    return faq_df

def find_best_match(user_question, faq_df, threshold=0.5):
    """
    사용자 질문과 FAQ의 각 질문 간 유사도를 계산하여 가장 유사한 항목을 반환하는 함수.
    유사도가 threshold 미만이면 None을 반환함.
    """
    best_match = None
    best_score = 0
    for idx, row in faq_df.iterrows():
        # row['question']를 문자열로 변환하여 비교
        score = similar(user_question, row['question'])
        if score > best_score:
            best_score = score
            best_match = row
    if best_score < threshold:
        return None, best_score
    return best_match, best_score

def chatbot():
    """
    FAQ 데이터를 기반으로 사용자 질문에 답변하는 간단한 챗봇 함수.
    """
    print("[INFO] FAQ 데이터를 로드합니다...")
    faq_df = get_faq_data()
    if faq_df is None or faq_df.empty:
        return

    print("[INFO] 챗봇을 시작합니다. (종료하려면 'exit' 입력)")
    while True:
        user_input = input("사용자 질문: ").strip()
        if user_input.lower() == "exit":
            print("[INFO] 챗봇을 종료합니다.")
            break

        match, score = find_best_match(user_input, faq_df)
        if match is not None:
            print(f"[답변] {match['answer']} (유사도: {score:.2f})")
        else:
            print("[답변] 죄송합니다. 해당 질문에 대한 답변을 찾을 수 없습니다.")

if __name__ == "__main__":
    try:
        chatbot()
    except Exception as e:
        print(f"[ERROR] 실행 중 오류 발생: {e}")