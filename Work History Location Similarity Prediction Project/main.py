from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from loadData import load_data


def predict_org_from_history(user_id):
    """
    USER_ID에 해당하는 ptbl_history 데이터를 불러와서,
    각 레코드의 HST_ORG_NM 컬럼 값과 ptbl_org 테이블의 ORG_FL_NM 값을 비교하여
    직접 매칭되는 값이 있으면 해당 ORG_FL_NM을 결과로 채택하고,
    매칭되는 값이 없을 경우 HST_ORG_NM과 ORG_FL_NM 간의 코사인 유사도를 계산하여
    가장 유사도가 높은 ORG_FL_NM을 예측 결과로 사용하는 함수입니다.
    """
    # 조직 데이터 로드 (org_query)
    org_query = "SELECT ORG_ID, ORG_FL_NM, ORG_PRNT_ID FROM ptbl_org"
    # 히스토리 데이터 로드 (history_query)
    history_query = f"""
    SELECT USER_ID, HST_ID, HST_ST_DT, HST_EDD_DT, HST_ORG_NM
    FROM ptbl_history
    WHERE USER_ID = '{user_id}'
    ORDER BY HST_ST_DT, HST_ID
    """
    print("[INFO] 데이터를 로드 중입니다...")
    org_data = load_data(org_query)
    history_data = load_data(history_query)

    if history_data.empty:
        print(f"[ERROR] USER_ID '{user_id}'에 대한 히스토리 데이터가 없습니다.")
        return

    # 결과를 저장할 컬럼 초기화
    history_data['Predicted_ORG'] = None

    # 각 히스토리 레코드에 대해 HST_ORG_NM을 기반으로 조직 매칭 또는 유사도 예측 수행
    for idx, row in history_data.iterrows():
        hst_org_nm = str(row['HST_ORG_NM']).strip()
        # 1. 직접 매칭: HST_ORG_NM 값이 org_data의 ORG_FL_NM과 정확히 일치하면 해당 ORG_FL_NM 사용
        direct_match = org_data[org_data['ORG_FL_NM'] == hst_org_nm]
        if not direct_match.empty:
            predicted = direct_match.iloc[0]['ORG_FL_NM']
            history_data.at[idx, 'Predicted_ORG'] = predicted
            print(f"[INFO] HST_ORG_NM '{hst_org_nm}'에 대한 직접 매칭 결과: {predicted}")
        else:
            # 2. 직접 매칭이 없는 경우, 코사인 유사도를 이용하여 예측 수행
            texts = [hst_org_nm] + org_data['ORG_FL_NM'].astype(str).tolist()
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(texts)
            cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            best_idx = cos_sim.argmax()
            predicted = org_data.iloc[best_idx]['ORG_FL_NM']
            history_data.at[idx, 'Predicted_ORG'] = predicted
            print(
                f"[INFO] HST_ORG_NM '{hst_org_nm}'에 대해 직접 매칭되지 않음. 유사도({cos_sim[best_idx]:.4f}) 기반 예측 결과: {predicted}")

    output_file = f"predicted_org_from_history_user_{user_id}.xlsx"
    history_data.to_excel(output_file, index=False)
    print(f"[INFO] 예측 결과가 '{output_file}' 파일로 저장되었습니다.")


if __name__ == "__main__":
    try:
        user_id = input("조회할 USER_ID를 입력하세요: ")
        predict_org_from_history(user_id)
        print("프로젝트 이름: Work History Organization Similarity Prediction Project")
    except Exception as e:
        print(f"[ERROR] 실행 중 오류 발생: {e}")