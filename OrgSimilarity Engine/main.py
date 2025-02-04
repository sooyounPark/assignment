from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import uuid
import re
from loadData import load_data


def get_new_org_id(org_data):
    """
    [요구사항 반영: ptbl_org 테이블의 ORG_ID 컬럼 데이터를 참고하여 새로운 ORG_ID 생성]
    기존 ORG_ID 중 "ORG_" 뒤에 숫자로 이루어진 패턴이 있다면,
    그 숫자들 중 최대값을 찾아 +1 한 값을 새 ORG_ID로 생성합니다.
    예) 기존 ORG_ID가 "ORG_0001", "ORG_0002"라면 새 ID는 "ORG_0003"이 됩니다.
    만약 숫자 패턴이 없으면 uuid를 활용하여 생성합니다.
    """
    max_num = -1
    pattern = re.compile(r'^ORG_(\d+)$')
    for org_id in org_data['ORG_ID']:
        if isinstance(org_id, str):
            m = pattern.match(org_id)
            if m:
                num = int(m.group(1))
                if num > max_num:
                    max_num = num
    if max_num >= 0:
        new_num = max_num + 1
        return f"ORG_{str(new_num).zfill(4)}"  # 4자리 포맷
    else:
        return "ORG_" + uuid.uuid4().hex[:8]


def get_new_org_prnt_cd(org_data):
    """
    [요구사항 반영: ptbl_org 테이블의 ORG_PRNT_CD 컬럼 데이터를 참고하여 새로운 ORG_PRNT_CD 생성]
    기존 ORG_PRNT_CD 중 "PRNT_CD_" 뒤에 숫자로 이루어진 패턴이 있다면,
    그 숫자들 중 최대값을 찾아 +1 한 값을 새 ORG_PRNT_CD로 생성합니다.
    예) 기존 ORG_PRNT_CD가 "PRNT_CD_0001", "PRNT_CD_0002"라면 새 코드는 "PRNT_CD_0003"이 됩니다.
    만약 숫자 패턴이 없으면 uuid를 활용하여 생성합니다.
    """
    max_num = -1
    pattern = re.compile(r'^PRNT_CD_(\d+)$')
    for prnt_cd in org_data['ORG_PRNT_CD']:
        if isinstance(prnt_cd, str):
            m = pattern.match(prnt_cd)
            if m:
                num = int(m.group(1))
                if num > max_num:
                    max_num = num
    if max_num >= 0:
        new_num = max_num + 1
        return f"PRNT_CD_{str(new_num).zfill(4)}"  # 4자리 포맷
    else:
        return "PRNT_CD_" + uuid.uuid4().hex[:8]


def get_parent_org_prnt_cd(org_data, parent_candidate):
    """
    부모 조직 후보(예: '전라남도 해남소방서')가 있을 경우, ptbl_org 테이블에서
    해당 조직명(ORG_FL_NM)이 일치하는 항목의 ORG_PRNT_CD를 반환합니다.
    만약 일치하는 항목이 없으면 새로운 ORG_PRNT_CD를 생성합니다.
    """
    matched = org_data[org_data['ORG_FL_NM'] == parent_candidate]
    if not matched.empty:
        # 기존에 등록된 부모 조직 코드 재사용
        return matched.iloc[0]['ORG_PRNT_CD']
    else:
        return get_new_org_prnt_cd(org_data)


def predict_new_org(org_name, org_data):
    """
    [요구사항 반영: 새로운 조직 정보 추측]
    유사도가 0.6 이하인 경우, 해당 조직명(HST_ORG_NM)을 바탕으로 새로운 조직 정보를 추측합니다.
    - ORG_ID: ptbl_org 테이블의 기존 ORG_ID 데이터를 참고하여 get_new_org_id 함수 사용
    - ORG_FL_NM: 입력된 org_name 그대로 사용
    - ORG_PRNT_ID: 조직명에 '소방서'가 포함되어 있으면, 해당 후보를 기준으로 부모 조직 ID를 생성(여기서는 UUID5 사용)
                      없으면 임의의 UUID 기반으로 생성합니다.
    - ORG_PRNT_CD: 조직명에 '소방서'가 포함되어 있으면, 해당 후보를 기준으로 기존 ptbl_org에서 부모 조직 코드를 재사용하거나
                   새로 생성합니다.
    """
    new_org_id = get_new_org_id(org_data)
    new_org_fl_nm = org_name

    parent_candidate = None
    if org_name and '소방서' in org_name:
        idx = org_name.find('소방서')
        # '소방서' 단어가 포함된 앞부분(예: "전라남도 해남소방서")을 부모 후보로 추출
        parent_candidate = org_name[:idx + len('소방서')]

    if parent_candidate:
        # 부모 조직 ID는 기존과 동일한 방식(UUID5)으로 생성 (부모 후보 문자열을 사용)
        org_prnt_id = "PARENT_" + uuid.uuid5(uuid.NAMESPACE_DNS, parent_candidate).hex[:8]
        # 부모 조직 코드는 ptbl_org 테이블의 데이터를 참고하여 생성 또는 재사용
        org_prnt_cd = get_parent_org_prnt_cd(org_data, parent_candidate)
    else:
        org_prnt_id = "PARENT_" + uuid.uuid4().hex[:8]
        org_prnt_cd = get_new_org_prnt_cd(org_data)

    return new_org_id, new_org_fl_nm, org_prnt_id, org_prnt_cd


def predict_org_and_new_entries(user_id):
    """
    [요구사항 반영: ptbl_org 테이블의 ORG_ID와 ORG_PRNT_CD를 참고하여
     유사도가 0.6 이하인 결과에 대해 새로운 조직 정보 예측]
    ptbl_history의 USER_ID에 해당하는 데이터를 로드한 후,
    각 레코드에 대해 HST_ORG_NM과 ptbl_org의 ORG_FL_NM을 비교하여
    직접 매칭 또는 TF-IDF 기반 코사인 유사도 예측을 수행합니다.
    그리고 예측된 유사도가 0.6 이하인 경우 ptbl_org 데이터를 참고하여 새로운 조직 정보를 추측합니다.
    """
    # 기존 조직 데이터 로드 (ORG_PRNT_CD 컬럼 포함)
    org_query = "SELECT ORG_ID, ORG_FL_NM, ORG_PRNT_ID, ORG_PRNT_CD FROM ptbl_org"
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

    new_org_entries = []
    history_data['Predicted_ORG'] = None
    history_data['Cosine_Similarity'] = None

    for idx, row in history_data.iterrows():
        hst_org_nm = str(row['HST_ORG_NM']).strip() if row['HST_ORG_NM'] is not None else ''
        # 1. 직접 매칭: HST_ORG_NM과 ORG_FL_NM이 정확히 일치하면 사용
        direct_match = org_data[org_data['ORG_FL_NM'] == hst_org_nm]
        if not direct_match.empty:
            predicted = direct_match.iloc[0]['ORG_FL_NM']
            history_data.at[idx, 'Predicted_ORG'] = predicted
            history_data.at[idx, 'Cosine_Similarity'] = 1.0
            print(f"[INFO] HST_ORG_NM '{hst_org_nm}'에 대한 직접 매칭 결과: {predicted}")
        else:
            # 2. 직접 매칭이 없으면 TF-IDF와 코사인 유사도를 이용해 예측 수행
            texts = [hst_org_nm] + org_data['ORG_FL_NM'].astype(str).tolist()
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(texts)
            cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            best_idx = cos_sim.argmax()
            best_sim = cos_sim[best_idx]
            predicted = org_data.iloc[best_idx]['ORG_FL_NM']
            history_data.at[idx, 'Predicted_ORG'] = predicted
            history_data.at[idx, 'Cosine_Similarity'] = best_sim

            if best_sim < 0.6:
                new_org_id, new_org_fl_nm, new_org_prnt_id, new_org_prnt_cd = predict_new_org(hst_org_nm, org_data)
                new_org_entries.append({
                    "ORG_ID": new_org_id,
                    "ORG_FL_NM": new_org_fl_nm,
                    "ORG_PRNT_ID": new_org_prnt_id,
                    "ORG_PRNT_CD": new_org_prnt_cd
                })
                print(f"[INFO] HST_ORG_NM '{hst_org_nm}'의 유사도({best_sim:.4f})가 0.6 이하입니다. "
                      f"예측된 새로운 조직 정보: ORG_ID={new_org_id}, ORG_FL_NM={new_org_fl_nm}, "
                      f"ORG_PRNT_ID={new_org_prnt_id}, ORG_PRNT_CD={new_org_prnt_cd}")
            else:
                print(f"[INFO] HST_ORG_NM '{hst_org_nm}'에 대해 직접 매칭되지 않음. "
                      f"유사도({best_sim:.4f}) 기반 예측 결과: {predicted}")

    output_file = f"predicted_org_from_history_user_{user_id}.xlsx"
    history_data.to_excel(output_file, index=False)
    print(f"[INFO] 예측 결과가 '{output_file}' 파일로 저장되었습니다.")

    if new_org_entries:
        new_org_df = pd.DataFrame(new_org_entries)
        new_org_file = f"new_org_entries_user_{user_id}.xlsx"
        new_org_df.to_excel(new_org_file, index=False)
        print(f"[INFO] 유사도가 0.6 이하인 새로운 조직 정보가 '{new_org_file}' 파일로 저장되었습니다.")
    else:
        print("[INFO] 유사도가 0.6 이하인 결과가 없습니다.")


if __name__ == "__main__":
    try:
        user_id = input("조회할 USER_ID를 입력하세요: ")
        predict_org_and_new_entries(user_id)
        print("프로젝트 이름: Work History Organization Similarity Prediction Project")
    except Exception as e:
        print(f"[ERROR] 실행 중 오류 발생: {e}")