import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from loadData import load_data

def classify_work_history(user_id):
    # 1. 데이터베이스에서 데이터 불러오기
    org_query = """
    SELECT ORG_FL_NM, ORG_TYP 
    FROM ptbl_org 
    WHERE ORG_TYP IS NOT NULL
    """
    org_df = load_data(org_query)

    history_query = f"""
    SELECT USER_ID, HST_ORG_NM 
    FROM ptbl_history
    WHERE USER_ID = '{user_id}'
    """
    history_df = load_data(history_query)

    if history_df.empty:
        print(f"[ERROR] USER_ID '{user_id}'에 대한 이력 데이터가 없습니다.")
        return

    # 2. 데이터 전처리
    org_df = org_df[org_df['ORG_TYP'].isin(['ORG', 'DEPT'])]

    if org_df.empty:
        print("[ERROR] 'ORG' 또는 'DEPT'로 분류된 조직 데이터가 없습니다.")
        return

    # Null 값 처리 및 공백 제거
    org_df['ORG_FL_NM'] = org_df['ORG_FL_NM'].fillna('').str.strip()
    org_df = org_df[org_df['ORG_FL_NM'] != '']

    if org_df.empty:
        print("[ERROR] ORG_FL_NM에 유효한 데이터가 없습니다.")
        return

    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer(stop_words=None)
    X = vectorizer.fit_transform(org_df['ORG_FL_NM'])
    y = org_df['ORG_TYP']

    # 3. 학습 및 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 랜덤포레스트 모델 학습
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 5. 모델 평가
    y_pred = clf.predict(X_test)
    print("[INFO] 모델 평가 결과:")
    print(classification_report(y_test, y_pred))

    # 6. 사용자 이력 데이터에 대한 예측
    history_df['HST_ORG_NM'] = history_df['HST_ORG_NM'].fillna('').str.strip()
    history_df = history_df[history_df['HST_ORG_NM'] != '']

    if history_df.empty:
        print("[ERROR] 유효한 조직명이 있는 이력 데이터가 없습니다.")
        return

    # 예측 수행
    new_X = vectorizer.transform(history_df['HST_ORG_NM'])
    predictions = clf.predict(new_X)

    # 7. 정교한 규칙 기반 분류 보정
    corrected_predictions = []
    for org_name, prediction in zip(history_df['HST_ORG_NM'], predictions):
        if '소방서' in org_name and ('센터' not in org_name and '119안전센터' not in org_name):
            corrected_predictions.append('ORG')  # '소방서'만 포함된 경우 ORG
        elif '센터' in org_name or '119안전센터' in org_name:
            corrected_predictions.append('DEPT')  # '센터'나 '119안전센터' 포함 시 DEPT
        else:
            corrected_predictions.append(prediction)  # 규칙에 해당하지 않으면 모델 예측 사용

    # 결과 저장
    result_df = pd.DataFrame({
        'USER_ID': history_df['USER_ID'],
        'HST_ORG_NM': history_df['HST_ORG_NM'],
        'Predicted_ORG_TYP': corrected_predictions
    })

    print("[INFO] 사용자 이력 분류 결과:")
    print(result_df)

    # 결과를 CSV로 저장
    output_file = f'classified_work_history_user_{user_id}.csv'
    result_df.to_csv(output_file, index=False)
    print(f"[INFO] 분류 결과가 '{output_file}'에 저장되었습니다.")


if __name__ == "__main__":
    try:
        user_id = input("조회할 USER_ID를 입력하세요: ")
        classify_work_history(user_id)
        print("프로젝트 이름: Work History Classification Project")
    except Exception as e:
        print(f"[ERROR] 실행 중 오류 발생: {e}")