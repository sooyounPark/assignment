import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy import create_engine
import urllib
import warnings
import numpy as np
from tensorflow.keras.models import load_model
import os
from beforeLearning import initialize_resources

warnings.filterwarnings("ignore")




# HST_TYP 추론 및 근무일수 계산
def train_and_infer_hst_typ(history_data):
    labeled_data = history_data[pd.notnull(history_data["HST_TYP"])]
    unlabeled_data = history_data[pd.isnull(history_data["HST_TYP"])]

    if labeled_data.empty:
        print("HST_TYP에 라벨링된 데이터가 없어 학습할 수 없습니다.")
        return history_data

    labeled_data["INCLUDE_LABEL"] = labeled_data["HST_TYP"].apply(
        lambda x: 0 if any(keyword in x for keyword in ["휴직", "복직", "승진"]) else 1
    )

    vectorizer = CountVectorizer()
    text_features = vectorizer.fit_transform(labeled_data["HST_TYP"].astype(str)).toarray()

    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(text_features, labeled_data["INCLUDE_LABEL"])

    if not unlabeled_data.empty:
        unlabeled_text_features = vectorizer.transform(unlabeled_data["HST_TYP"].astype(str)).toarray()
        unlabeled_data["INCLUDE_LABEL"] = rf.predict(unlabeled_text_features)

    history_data = pd.concat([labeled_data, unlabeled_data], axis=0)
    history_data["INFERRED_HST_TYP"] = history_data["INCLUDE_LABEL"].apply(lambda x: "포함" if x == 1 else "제외")

    # 날짜 형식 변환 추가
    history_data["HST_ST_DT"] = pd.to_datetime(history_data["HST_ST_DT"], errors="coerce")
    history_data["HST_EDD_DT"] = pd.to_datetime(history_data["HST_EDD_DT"], errors="coerce")

    # 변환 후 데이터 확인
    print("[DEBUG] 날짜 변환 후 데이터 확인:")
    print(history_data[["HST_ST_DT", "HST_EDD_DT"]].dtypes)

    # 근무일수 계산
    history_data["근무일수"] = (history_data["HST_EDD_DT"] - history_data["HST_ST_DT"]).dt.days.fillna(0)
    history_data["제외 근무일수"] = history_data.apply(
        lambda row: row["근무일수"] if row["INFERRED_HST_TYP"] == "제외" else 0, axis=1
    )

    return history_data
def unify_organization_history(history_data):
    """
    상위 조직(ORG_PRNT_ID)을 기준으로 동일 조직 여부를 통합 처리
    """
    # 동일 조직 여부 확인: ORG_PRNT_ID 기준
    history_data["UNIFIED_ORG"] = history_data.apply(
        lambda row: row["ORG_FL_NM_PARENT"] if pd.notnull(row["ORG_FL_NM_PARENT"]) else row["HST_ORG_NM"],
        axis=1
    )

    # 동일 조직 그룹화 처리
    history_data["GROUPED_ORG"] = history_data.groupby("ORG_PRNT_ID")["UNIFIED_ORG"].transform("first")

    return history_data
# 관서/센터 추론 함수 (Random Forest)
def train_and_infer_hst_location_with_ml(history_data, org_data):
    # 데이터 확인
    print("[DEBUG] org_data 샘플 데이터 확인:")
    print(org_data.head())

    # 병합된 데이터에서 ORG_PRNT_ID 유효성 확인
    if history_data["ORG_PRNT_ID"].isna().any():
        print("[ERROR] ORG_PRNT_ID 없는 데이터가 있습니다.")
        print(history_data[history_data["ORG_PRNT_ID"].isna()])
        raise ValueError("ORG_PRNT_ID NaN인 데이터를 처리해야 합니다.")

    # 라벨링된 데이터: ORG_FL_NM 기반
    labeled_data = org_data[pd.notnull(org_data["ORG_FL_NM"])]
    if labeled_data.empty:
        print("관서/센터 분류를 위한 학습 데이터가 없습니다.")
        return history_data

    labeled_data["LOCATION_LABEL"] = labeled_data["ORG_FL_NM"].apply(
        lambda x: 1 if "소방서" in x else 0
    )

    vectorizer = CountVectorizer()
    text_features = vectorizer.fit_transform(labeled_data["ORG_FL_NM"].astype(str)).toarray()

    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(text_features, labeled_data["LOCATION_LABEL"])


    # 관서/센터 추론
    unlabeled_data = history_data[pd.isnull(history_data["ORG_FL_NM"])]
    if not unlabeled_data.empty:
        unlabeled_text_features = vectorizer.transform(unlabeled_data["HST_ORG_NM"].astype(str)).toarray()
        unlabeled_data["LOCATION_LABEL"] = rf.predict(unlabeled_text_features)
        unlabeled_data["INFERRED_LOCATION"] = unlabeled_data["LOCATION_LABEL"].apply(lambda x: "관서" if x == 1 else "센터")

    labeled_data = history_data[pd.notnull(history_data["ORG_FL_NM"])]
    labeled_data["INFERRED_LOCATION"] = labeled_data["ORG_FL_NM"].apply(lambda x: "관서" if "소방서" in x else "센터")

    history_data = pd.concat([labeled_data, unlabeled_data], axis=0)
    return history_data
# 이력 분석 함수
def analyze_history_with_location(history_data, org_data, user_code):
    user_history = history_data[history_data["USER_ID"] == user_code].copy()
    if user_history.empty:
        print(f"사용자 코드 '{user_code}'에 대한 이력이 없습니다.")
        return None

    user_history = user_history[user_history["INFERRED_HST_TYP"] == "포함"]
    if user_history.empty:
        print(f"'{user_code}' 사용자의 유효한 근무 이력이 없습니다.")
        return None

    user_history = user_history.sort_values(by="HST_ST_DT").reset_index(drop=True)
    user_history["HST_DURATION"] = (user_history["HST_EDD_DT"] - user_history["HST_ST_DT"]).dt.days.fillna(0)

    user_history = user_history.merge(
        org_data[["ORG_ID", "ORG_FL_NM"]],
        left_on="ORG_PRNT_ID",
        right_on="ORG_ID",
        how="left",
        suffixes=("", "_PARENT")
    )

    def assign_labels_by_org(df):
        """
        org_id와 org_prnt_id를 기반으로 '현 조직', '전 조직', '전전 조직'을 구분합니다.
        - 센터인 경우 org_id를 기준으로 비교.
        - 관서인 경우 org_prnt_id를 기준으로 비교.
        """
        # 데이터프레임을 정렬: HST_ST_DT 기준으로 정렬
        df = df.sort_values(by="HST_ST_DT").reset_index(drop=True)

        # 레이블을 저장할 리스트
        labels = [""] * len(df)

        # 기본적으로 마지막 행은 항상 '현 조직'
        labels[-1] = "현 조직"

        # 역순으로 탐색하여 이전 행과 비교
        for i in range(len(df) - 2, -1, -1):
            current_row = df.iloc[i]
            next_row = df.iloc[i + 1]

            # 센터인 경우: org_id 비교
            if current_row["INFERRED_LOCATION"] == "센터":
                if current_row["ORG_ID"] != next_row["ORG_ID"]:
                    labels[i] = "전 조직" if labels[i + 1] == "현 조직" else "전전 조직"
                else:
                    labels[i] = labels[i + 1]  # 동일하면 레이블 유지

            # 관서인 경우: org_prnt_id 비교
            elif current_row["INFERRED_LOCATION"] == "관서":
                if current_row["ORG_PRNT_ID"] != next_row["ORG_PRNT_ID"]:
                    labels[i] = "전 조직" if labels[i + 1] == "현 조직" else "전전 조직"
                else:
                    labels[i] = labels[i + 1]  # 동일하면 레이블 유지
            else:
                # 예상치 못한 경우 '미분류'로 처리
                labels[i] = "미분류"

        return labels

    user_history["구분"] = assign_labels_by_org(user_history)
    user_history["이동"] = user_history["HST_ORG_NM"].shift(1) + " → " + user_history["HST_ORG_NM"]
    user_history["이동"] = user_history["이동"].fillna("초기 조직")

    return user_history
def reorder_history_by_org(history_data):
    """
    HST_ST_DT, HST_ID 기준 정렬 후 ORG_FL_NM 유사도 기반으로 재정렬
    """
    # Step 1: HST_ST_DT와 HST_ID로 1차 정렬
    history_data = history_data.sort_values(by=["HST_ST_DT", "HST_ID"]).reset_index(drop=True)

    # Step 2: ORG_FL_NM 기반 유사도 분석 (NULL 값 처리)
    history_data["ORG_FL_NM"] = history_data["ORG_FL_NM"].fillna("")
    vectorizer = CountVectorizer()
    org_vectors = vectorizer.fit_transform(history_data["ORG_FL_NM"])

    # Cosine Similarity를 이용한 유사도 계산
    similarity_matrix = cosine_similarity(org_vectors)

    # 순서 재정렬: 가장 유사한 조직 순으로 정렬
    reordered_indices = similarity_matrix.sum(axis=1).argsort()
    history_data = history_data.iloc[reordered_indices].reset_index(drop=True)

    return history_data
def train_and_infer_hst_location_with_rules(history_data, org_data):
    """
    etbl_org 테이블을 활용하여 관서/센터를 추론 (규칙 기반)
    """
    # 1. HST_ORG_NM → ORG_ID 매핑
    history_data = history_data.merge(
        org_data[["ORG_ID", "ORG_FL_NM", "ORG_PRNT_ID"]],
        left_on="HST_ORG_NM",
        right_on="ORG_FL_NM",
        how="left"
    )

    # 2. 상위 조직 매칭
    history_data = history_data.merge(
        org_data[["ORG_ID", "ORG_FL_NM", "ORG_TYP", "ORG_PRNT_YN"]],
        left_on="ORG_PRNT_ID",
        right_on="ORG_ID",
        how="left",
        suffixes=("", "_PARENT")
    )

    print("[DEBUG] 상위 조직 매칭 결과:")
    print(history_data[["ORG_ID", "ORG_FL_NM", "ORG_PRNT_ID", "ORG_FL_NM_PARENT", "ORG_TYP", "ORG_TYP_PARENT"]].drop_duplicates())

    # 3. 관서/센터 분류
    def infer_location(row):
        # 부모 조직 여부 확인
        if row["ORG_PRNT_YN_PARENT"] == "Y" and row["ORG_TYP_PARENT"] == "관서":
            return "관서"
        # 조직 이름에 "소방서" 포함 여부
        elif "소방서" in str(row["ORG_FL_NM_PARENT"]):
            return "관서"
        # 특정 조직 유형에 따른 분류
        elif row["ORG_TYP_PARENT"] in ["특정_관서_유형"]:  # 예: 관서 유형이 명확한 경우
            return "관서"
        else:
            return "센터"

    history_data["INFERRED_LOCATION"] = history_data.apply(infer_location, axis=1)
    return history_data

def redefine_grouped_organization_labels(history_data):
    """
    GROUPED_ORG 기준으로 '구분' 값 재정의
    """
    def assign_labels(group):
        # 그룹 내 정렬
        group = group.sort_values(by="HST_ST_DT").reset_index(drop=True)

        # 구분 레이블 설정
        labels = ["전전 조직"] * max(0, len(group) - 2) + ["전 조직"] * min(1, len(group) - 1) + ["현 조직"]
        group["구분"] = labels[-len(group):]
        return group

    # GROUPED_ORG 기준으로 그룹화 후 레이블 재정의
    history_data = history_data.groupby("GROUPED_ORG", group_keys=False).apply(assign_labels)
    return history_data
def fill_missing_org_prnt_cd(history_data, org_data):
    """
    ORG_PRNT_ID NaN인 데이터를 추론하여 채웁니다.
    """
    # Vectorizer 초기화 및 org_data 벡터화
    vectorizer = CountVectorizer()
    org_data_vectors = vectorizer.fit_transform(org_data["ORG_FL_NM"].astype(str))

    def infer_org_prnt_cd(row):
        """
        조직명을 기반으로 ORG_PRNT_CD 추론
        """
        # 행 벡터화
        row_vector = vectorizer.transform([row["HST_ORG_NM"]])
        similarities = cosine_similarity(row_vector, org_data_vectors)[0]

        # 가장 높은 유사도를 가진 조직 찾기
        max_similarity = similarities.max()
        matched_index = similarities.argmax()
        matched_org = org_data.iloc[matched_index]

        # 디버깅 정보 출력
        print("[DEBUG] 현재 처리 중인 행(row):")
        print(row)
        print(f"[DEBUG] 최대 유사도: {max_similarity}")
        print("[DEBUG] 매칭된 조직 데이터:")
        print(matched_org)

        # 임계값(0.7) 이상인 경우 매칭
        if max_similarity >= 0.7 and not matched_org.empty:
            if matched_org["ORG_PRNT_ID"] == "1":
                return matched_org["ORG_ID"]
            else:
                return matched_org["ORG_PRNT_ID"]




        # 추론 불가한 경우
        print("[DEBUG] 추론 불가.")
        return None

    def debug_infer_org_prnt_cd(row):
        print(f"[DEBUG] 현재 처리 중인 row의 ORG_PRNT_ID: {row['ORG_PRNT_ID']}")
        print(f"[DEBUG] pd.isna(row['ORG_PRNT_ID']): {pd.isna(row['ORG_PRNT_ID'])}")
        print(f"[DEBUG] row['ORG_PRNT_ID'] == None: {row['ORG_PRNT_ID'] is None}")
        return infer_org_prnt_cd(row)

    history_data["INFERRED_ORG_PRNT_ID"] = history_data.apply(
        lambda row: debug_infer_org_prnt_cd(row) if pd.isna(row["ORG_PRNT_ID"]) else row["ORG_PRNT_ID"],
        axis=1
    )

    # NaN 값 확인
    print("[DEBUG] 추론된 ORG_PRNT_ID 결과:")
    print(history_data[history_data["INFERRED_ORG_PRNT_ID"].isna()])
    print(history_data[history_data["ORG_PRNT_ID"].isna()])

    # NaN 값을 기본값(1)로 채움
    history_data["ORG_PRNT_ID"] = history_data["INFERRED_ORG_PRNT_ID"].fillna("1")
    history_data.drop(columns=["INFERRED_ORG_PRNT_ID"], inplace=True)

    return history_data
def fill_missing_org_id(history_data, org_data):
    """
    ORG_ID가 NaN인 데이터를 추론하여 채웁니다.
    """
    # Vectorizer 초기화 및 org_data 벡터화
    vectorizer = CountVectorizer()
    org_data_vectors = vectorizer.fit_transform(org_data["ORG_FL_NM"].astype(str))

    def infer_org_id(row):
        """
        조직명을 기반으로 ORG_ID 추론
        """
        # HST_ORG_NM 기반으로 ORG_ID 추론
        row_vector = vectorizer.transform([row["HST_ORG_NM"]])
        similarities = cosine_similarity(row_vector, org_data_vectors)[0]

        # 가장 높은 유사도를 가진 조직 찾기
        max_similarity = similarities.max()
        matched_index = similarities.argmax()
        matched_org = org_data.iloc[matched_index]

        # 임계값(0.7) 이상인 경우 ORG_ID 반환
        if max_similarity >= 0.7:
            return matched_org["ORG_ID"]

        # 기본값 반환: "(임시)본부"
        return "TEMP_ORG_ID"

    # 누락된 ORG_ID를 채움
    history_data["ORG_ID"] = history_data.apply(
        lambda row: infer_org_id(row) if pd.isna(row["ORG_ID"]) else row["ORG_ID"], axis=1
    )

    return history_data
    def debug_infer_org_id(row):
        print(f"[DEBUG] 현재 처리 중인 row의 ORG_ID: {row['ORG_ID']}")
        print(f"[DEBUG] pd.isna(row['ORG_ID']): {pd.isna(row['ORG_ID'])}")
        return infer_org_id(row)

    history_data["INFERRED_ORG_ID"] = history_data.apply(
        lambda row: debug_infer_org_id(row) if pd.isna(row["ORG_ID"]) else row["ORG_ID"],
        axis=1
    )
    history_data["INFERRED_ORG_FL_NM"] = history_data.apply(
        lambda row: debug_infer_org_id(row) if pd.isna(row["ORG_FL_NM"]) else row["ORG_FL_NM"],
        axis=1
    )


    # NaN 값 확인
    print("[DEBUG] 추론된 ORG_ID 결과:")
    print(history_data[history_data["INFERRED_ORG_ID"].isna()])

    # NaN 값을 None 또는 기본 값으로 채움
    history_data["ORG_ID"] = history_data["INFERRED_ORG_ID"].fillna("1")
    history_data["ORG_FL_NM"] = history_data["INFERRED_ORG_FL_NM"].fillna("(임시)본부")
    history_data.drop(columns=["INFERRED_ORG_ID"], inplace=True)
    history_data.drop(columns=["INFERRED_ORG_FL_NM"], inplace=True)

    return history_data

def unify_organization_history(history_data):
    """
    상위 조직(ORG_PRNT_CD)을 기준으로 동일 조직 여부를 통합 처리
    """
    # 동일 조직 여부 확인: ORG_PRNT_CD 기준
    def determine_unified_org(row):
        if pd.notnull(row["ORG_FL_NM_PARENT"]):
            return row["ORG_FL_NM_PARENT"]
        elif pd.notnull(row["HST_ORG_NM"]):
            return row["HST_ORG_NM"]
        else:
            return "미확인 조직"

    # UNIFIED_ORG 컬럼 생성
    history_data["UNIFIED_ORG"] = history_data.apply(determine_unified_org, axis=1)

    # GROUPED_ORG 생성: ORG_PRNT_CD가 존재하지 않으면 NaN 처리 방지
    if "ORG_PRNT_ID" in history_data.columns:
        history_data["GROUPED_ORG"] = history_data.groupby("ORG_PRNT_ID")["UNIFIED_ORG"].transform("first")
    else:
        history_data["GROUPED_ORG"] = history_data["UNIFIED_ORG"]

    return history_data

def fill_missing_org_id_and_name(history_data, org_data):
    """
    ORG_ID와 ORG_FL_NM이 NaN인 데이터를 추론하여 채웁니다.
    """
    # Vectorizer 초기화 및 org_data 벡터화
    vectorizer = CountVectorizer()
    org_data_vectors = vectorizer.fit_transform(org_data["ORG_FL_NM"].astype(str))

    def infer_org_info(row):
        """
        조직명을 기반으로 ORG_ID와 ORG_FL_NM 추론
        """
        # HST_ORG_NM 기반으로 ORG_ID와 ORG_FL_NM 추론
        row_vector = vectorizer.transform([row["HST_ORG_NM"]])
        similarities = cosine_similarity(row_vector, org_data_vectors)[0]

        # 가장 높은 유사도를 가진 조직 찾기
        max_similarity = similarities.max()
        matched_index = similarities.argmax()
        matched_org = org_data.iloc[matched_index]

        # 임계값(0.7) 이상인 경우 ORG_ID와 ORG_FL_NM 반환
        if max_similarity >= 0.7:
            return matched_org["ORG_ID"], matched_org["ORG_FL_NM"]

        # 기본값 반환: TEMP_ORG_ID와 TEMP_ORG_NAME
        return "TEMP_ORG_ID", "(임시)본부"

    # 누락된 ORG_ID와 ORG_FL_NM를 채움
    def fill_row(row):
        if pd.isna(row["ORG_ID"]) or pd.isna(row["ORG_FL_NM"]):
            inferred_id, inferred_name = infer_org_info(row)
            row["ORG_ID"] = inferred_id
            row["ORG_FL_NM"] = inferred_name
        return row

    # Apply the inference function row-wise
    history_data = history_data.apply(fill_row, axis=1)

    return history_data


if __name__ == "__main__":
    try:
        print("[INFO] 데이터를 로드 중입니다...")

        # 데이터 로드 및 사전 처리
        org_data, history_data, user_data, label_encoder, vectorizer = initialize_resources()

        print("[INFO] 데이터 로드 완료!")
        print("조직 데이터 예시:")
        print(org_data.head())
        print("이력 데이터 예시:")
        print(history_data.head())
        print("사용자 데이터 예시:")
        print(user_data.head())

        # 디지털 코드와 기준 입력
        user_code = input("디지털 코드를 입력하세요 (USER_ID): ").strip()

        # 사용자 데이터 필터링
        print("사용자 데이터 필터링 중입니다...")
        user_history = history_data[history_data["USER_ID"] == user_code]
        if user_history.empty:
            print(f"사용자 코드 '{user_code}'에 대한 이력이 없습니다.")
            exit()

        # 병합 작업
        print("병합 작업 중입니다...")
        user_history = user_history.merge(
            org_data[["ORG_ID", "ORG_FL_NM", "ORG_PRNT_ID"]],
            left_on="HST_ORG_NM",
            right_on="ORG_FL_NM",
            how="left"
        )

        # ORG_PRNT_ID가 '1'인 경우 ORG_ID로 값 설정
        user_history["ORG_PRNT_ID"] = user_history.apply(
            lambda row: row["ORG_ID"] if row["ORG_PRNT_ID"] == "1" else row["ORG_PRNT_ID"],
            axis=1
        )

        # ORG_ID 및 ORG_FL_NM 없는 데이터 처리
        if user_history["ORG_ID"].isna().any() or user_history["ORG_FL_NM"].isna().any():
            print("[INFO] ORG_ID 또는 ORG_FL_NM이 없는 데이터를 처리 중...")
            user_history = fill_missing_org_id_and_name(user_history, org_data)

        # ORG_PRNT_ID 없는 데이터 처리
        if user_history["ORG_PRNT_ID"].isna().any():
            print("[ERROR] ORG_PRNT_ID 없는 데이터가 있습니다. 추론 중입니다...")
            user_history = fill_missing_org_prnt_cd(user_history, org_data)

        # TEMP_ORG_ID가 있는 데이터 확인
        temp_org_data = user_history[user_history["ORG_ID"] == "TEMP_ORG_ID"]
        if not temp_org_data.empty:
            print("[WARNING] 일부 데이터의 ORG_ID가 'TEMP_ORG_ID'로 설정되었습니다. 확인이 필요합니다.")
            print(temp_org_data)

        # 상위 조직 병합
        user_history = user_history.merge(
            org_data[["ORG_ID", "ORG_FL_NM"]],
            left_on="ORG_PRNT_ID",
            right_on="ORG_ID",
            how="left",
            suffixes=("", "_PARENT")
        )

        # 정렬 및 추론 로직 실행
        print("이력 정렬 중입니다...")
        user_history = reorder_history_by_org(user_history)
        print("이력 정렬 완료.\n")

        print("조직 통합 처리 중입니다...")
        user_history = unify_organization_history(user_history)
        print("조직 통합 처리 완료.\n")

        print("관서/센터 추론 중입니다...")
        user_history = train_and_infer_hst_location_with_ml(user_history, org_data)
        print("관서/센터 추론 완료.\n")

        print("HST_TYP 추론 중입니다...")
        user_history = train_and_infer_hst_typ(user_history)
        print("HST_TYP 추론 완료.\n")

        print("구분 값 재정의 중입니다...")
        user_history = redefine_grouped_organization_labels(user_history)
        print("구분 값 재정의 완료.\n")

        # 기준별 이력 분석
        result = analyze_history_with_location(user_history, org_data, user_code)
        if result is not None:
            print("\n사용자 이력 분석 결과:")
            print(result[[
                "구분", "HST_ST_DT", "HST_EDD_DT", "HST_ORG_NM", "ORG_FL_NM_PARENT",
                "ORG_ID_PARENT", "INFERRED_LOCATION", "INFERRED_HST_TYP",
                "근무일수", "제외 근무일수", "이동", "GROUPED_ORG"
            ]])

            # 결과 저장
            output_filename = f"analyzed_history_{user_code}.xlsx"

            # 파일 존재 여부 확인 및 덮어쓰기 처리
            if os.path.exists(output_filename):
                print(f"[INFO] '{output_filename}' 파일이 이미 존재합니다. 덮어쓰기 진행 중...")
            else:
                print(f"[INFO] '{output_filename}' 파일을 새로 생성합니다.")

            # 엑셀 파일 저장
            result.to_excel(output_filename, index=False)
            print(f"\n결과가 '{output_filename}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")