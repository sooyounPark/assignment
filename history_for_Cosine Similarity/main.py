from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from loadData import load_data

def classify_work_history_for_user(user_id, start_label="근무이력", end_label="근무이력"):
    # 데이터 로드
    history_query = f"""
     SELECT USER_ID, HST_ID, HST_ST_DT, HST_EDD_DT, HST_ORG_NM, HST_TYP
     FROM ptbl_history
     WHERE USER_ID = '{user_id}'
     ORDER BY HST_ST_DT, HST_ID
     """
    print("[INFO] 데이터를 로드 중입니다...")
    history_data = load_data(history_query)

    if history_data.empty:
        print(f"[ERROR] USER_ID '{user_id}'에 대한 데이터가 없습니다.")
        return

    # 입력값 검증
    if not any(history_data['HST_TYP'].str.contains(start_label, na=False)):
        print(f"[ERROR] 시작 구분 이력 구분값 '{start_label}'이 데이터에 존재하지 않습니다.")
        return

    if not any(history_data['HST_TYP'].str.contains(end_label, na=False)):
        print(f"[ERROR] 종료 구분 이력 구분값 '{end_label}'이 데이터에 존재하지 않습니다.")
        return

    # 기본 라벨링 및 날짜 변환
    history_data['WORK_LABEL'] = 1  # 기본값: 근무 이력
    label_encoder = LabelEncoder()
    history_data['HST_TYP_ENCODED'] = label_encoder.fit_transform(history_data['HST_TYP'])

    def safe_datetime_to_timestamp(date):
        try:
            return int(pd.to_datetime(date).timestamp())
        except Exception:
            return np.nan

    history_data['HST_ST_DT'] = history_data['HST_ST_DT'].apply(safe_datetime_to_timestamp)
    history_data['HST_EDD_DT'] = history_data['HST_EDD_DT'].apply(safe_datetime_to_timestamp)

    # 시작/종료 구분값 처리
    user_data = history_data.copy()
    user_data['HST_TYP_CLEAN'] = user_data['HST_TYP'].apply(lambda x: x.strip().lower() if isinstance(x, str) else "")

    start_idx = user_data[user_data['HST_TYP_CLEAN'].str.contains(start_label)].index
    end_idx = user_data[user_data['HST_TYP_CLEAN'].str.contains(end_label)].index

    if not start_idx.empty and not end_idx.empty:
        start = start_idx[0]
        end = end_idx[0]
        if start < end:
            history_data.loc[start:end-1, 'WORK_LABEL'] = 0
            history_data.loc[end, 'WORK_LABEL'] = 1
        else:
            print("[WARNING] 종료 구분값이 시작 구분값보다 앞에 있습니다.")

    # 기간 특성 추가
    history_data['DURATION'] = history_data['HST_EDD_DT'] - history_data['HST_ST_DT']
    features = ['HST_ST_DT', 'HST_EDD_DT', 'HST_TYP_ENCODED', 'DURATION']
    X = history_data[features].fillna(0)
    y = history_data['WORK_LABEL']


    # 클래스 분포 확인
    print("[DEBUG] WORK_LABEL 클래스 분포:")
    print(history_data['WORK_LABEL'].value_counts())

    if len(history_data['WORK_LABEL'].unique()) <= 1:
        print("[ERROR] WORK_LABEL 데이터가 단일 클래스입니다.")
        return

    # 데이터 증강 (SMOTE)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    print("[INFO] Train/Test 데이터 분포:")
    print("Train 데이터 분포:\n", pd.Series(y_train).value_counts())
    print("Test 데이터 분포:\n", pd.Series(y_test).value_counts())

    # 랜덤 포레스트 모델 학습
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight={0: 1, 1: 2},
        random_state=42
    )
    model.fit(X_train, y_train)

    # 모델 평가
    y_pred = model.predict(X_test)
    print("[INFO] 분류 결과:")
    print("정확도:", accuracy_score(y_test, y_pred))
    print("\n분류 보고서:\n", classification_report(y_test, y_pred))

    # 특성 중요도 출력
    feature_importances = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\n[INFO] 특성 중요도:\n", feature_importances)

    # 결과 저장
    history_data['PREDICTED_WORK_LABEL'] = model.predict(X)
    output_file = f"classified_history_user_{user_id}.xlsx"
    history_data.to_excel(output_file, index=False)
    print(f"[INFO] 결과가 '{output_file}'에 저장되었습니다.")


if __name__ == "__main__":
    try:
        user_id = input("조회할 USER_ID를 입력하세요: ")
        start_label = input("시작 구분 이력 구분값: ")
        end_label = input("종료 구분 이력 구분값: ")

        classify_work_history_for_user(user_id=user_id, start_label=start_label, end_label=end_label)
    except Exception as e:
        print(f"[ERROR] 실행 중 오류 발생: {e}")