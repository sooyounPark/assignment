import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 비인터랙티브 백엔드를 사용하여 HTTP 429 에러 회피
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from loadData import load_data
from datetime import datetime, date

# 한글 폰트 및 마이너스 기호 설정
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

def classify_work_history(user_id):
    ##############################
    # 1. 데이터 불러오기 (조직 데이터 및 해당 사용자 이력 데이터)
    ##############################
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

    ##############################
    # 2. 데이터 전처리
    ##############################
    org_df = org_df[org_df['ORG_TYP'].isin(['ORG', 'DEPT'])].copy()
    if org_df.empty:
        print("[ERROR] 'ORG' 또는 'DEPT'로 분류된 조직 데이터가 없습니다.")
        return

    org_df['ORG_FL_NM'] = org_df['ORG_FL_NM'].fillna('').str.strip()
    org_df = org_df[org_df['ORG_FL_NM'] != ''].copy()
    if org_df.empty:
        print("[ERROR] ORG_FL_NM에 유효한 데이터가 없습니다.")
        return

    vectorizer = TfidfVectorizer(stop_words=None)
    X = vectorizer.fit_transform(org_df['ORG_FL_NM'])
    y = org_df['ORG_TYP']

    ##############################
    # 3. 학습/테스트 데이터 분할
    ##############################
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ##############################
    # 4. Random Forest 모델 학습
    ##############################
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    ##############################
    # 5. 모델 평가 및 시각화 자료 생성
    ##############################
    y_pred = clf.predict(X_test)
    print(f"[INFO] USER_ID '{user_id}' - 모델 평가 결과:")
    print(classification_report(y_test, y_pred))

    # (a) 혼동 행렬 시각화
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Figure 1. Confusion Matrix (USER_ID: {user_id})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_user_{user_id}.png')
    plt.close()

    # (b) 특징 중요도 시각화
    feature_names = vectorizer.get_feature_names_out()
    importances = clf.feature_importances_
    indices = importances.argsort()[::-1][:20]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features)
    plt.title(f'Figure 2. Top 20 Feature Importances (USER_ID: {user_id})')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'feature_importances_user_{user_id}.png')
    plt.close()

    # (c) ROC Curve 및 AUC 시각화
    y_test_binary = (y_test == 'DEPT').astype(int)
    dept_index = list(clf.classes_).index('DEPT')
    y_score = clf.predict_proba(X_test)[:, dept_index]
    fpr, tpr, _ = roc_curve(y_test_binary, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Figure 3. ROC Curve (USER_ID: {user_id})')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'roc_curve_user_{user_id}.png')
    plt.close()

    # (d) t-SNE 시각화
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_train.toarray())
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_train, alpha=0.7)
    plt.title(f'Figure 4. t-SNE Visualization (USER_ID: {user_id})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title='ORG_TYP')
    plt.tight_layout()
    plt.savefig(f'tsne_visualization_user_{user_id}.png')
    plt.close()

    # (e) 사용자 이력 예측 분포 비교
    history_df['HST_ORG_NM'] = history_df['HST_ORG_NM'].fillna('').str.strip()
    history_df = history_df[history_df['HST_ORG_NM'] != ''].copy()
    if not history_df.empty:
        new_X = vectorizer.transform(history_df['HST_ORG_NM'])
        preds = clf.predict(new_X)

        # 도메인 규칙 기반 보정
        corrected = []
        for name, p in zip(history_df['HST_ORG_NM'], preds):
            if '소방서' in name and '센터' not in name:
                corrected.append('ORG')
            elif '센터' in name:
                corrected.append('DEPT')
            else:
                corrected.append(p)

        result_df = pd.DataFrame({
            'USER_ID': history_df['USER_ID'],
            'HST_ORG_NM': history_df['HST_ORG_NM'],
            'Predicted': preds,
            'Corrected': corrected
        })

        print(f"[INFO] USER_ID '{user_id}' - 이력 분류 결과:")
        print(result_df)

        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        result_df['Predicted'].value_counts().plot(kind='bar')
        plt.title('Model Predicted')

        plt.subplot(1,2,2)
        result_df['Corrected'].value_counts().plot(kind='bar')
        plt.title('Corrected')
        plt.tight_layout()
        plt.savefig(f'prediction_distribution_user_{user_id}.png')
        plt.close()

        # CSV 저장
        result_df.to_csv(f'classified_work_history_user_{user_id}.csv', index=False)
    else:
        print("[WARN] 사용자 이력 데이터가 없어 분포 비교를 건너뜁니다.")

def estimate_working_days(user_id):
    ##############################
    # 이력 구분별 실제 근무 일수 계산
    ##############################
    history_query = f"""
    SELECT HST_TYP, HST_ST_DT, HST_EDD_DT
    FROM ptbl_history
    WHERE USER_ID = '{user_id}'
    """
    history_df = load_data(history_query)
    if history_df.empty:
        print(f"[ERROR] USER_ID '{user_id}'에 대한 이력 데이터가 없습니다.")
        return

    history_df['HST_ST_DT'] = pd.to_datetime(history_df['HST_ST_DT'], errors='coerce')
    history_df['HST_EDD_DT'] = pd.to_datetime(history_df['HST_EDD_DT'], errors='coerce')
    history_df = history_df.dropna(subset=['HST_ST_DT','HST_EDD_DT'])
    if history_df.empty:
        print("[ERROR] 유효한 시작일 또는 종료일이 있는 이력 데이터가 없습니다.")
        return

    # 종료일 포함 근무 일수 계산
    history_df['WORK_DAYS'] = (history_df['HST_EDD_DT'] - history_df['HST_ST_DT']).dt.days + 1

    summary = history_df.groupby('HST_TYP')['WORK_DAYS'].sum().reset_index()
    summary.columns = ['HST_TYP', 'TOTAL_WORK_DAYS']

    print(f"[INFO] USER_ID '{user_id}' - 실제 근무 일수 요약:")
    print(summary)

    # CSV 저장
    summary.to_csv(f'actual_working_days_user_{user_id}.csv', index=False)

def estimate_working_days(user_id):
    ##############################
    # 이력 구분별 실제 근무 일수 계산 (종료일 추론 포함)
    ##############################
    history_query = f"""
    SELECT HST_TYP, HST_ST_DT, HST_ED_DT
    FROM ptbl_history
    WHERE USER_ID = '{user_id}'
    """
    history_df = load_data(history_query)
    if history_df.empty:
        print(f"[ERROR] USER_ID '{user_id}'에 대한 이력 데이터가 없습니다.")
        return

    # 날짜 형식 변환
    history_df['HST_ST_DT'] = pd.to_datetime(history_df['HST_ST_DT'], errors='coerce')
    history_df['HST_ED_DT'] = pd.to_datetime(history_df['HST_ED_DT'], errors='coerce')
    history_df = history_df.dropna(subset=['HST_ST_DT'])
    if history_df.empty:
        print("[ERROR] 유효한 시작일이 있는 이력 데이터가 없습니다.")
        return

    # 시작일 기준 정렬
    history_df = history_df.sort_values('HST_ST_DT').reset_index(drop=True)

    # 종료일이 없는 행 추론: 다음 시작일 - 1일, 마지막은 오늘 날짜
    # shift(-1)로 다음 행의 HST_ST_DT를 가져온 뒤 하루 빼기
    next_st = history_df['HST_ST_DT'].shift(-1)
    inferred_ed = next_st - pd.Timedelta(days=1)
    # 마지막 행의 inferred_ed는 NaT이므로 오늘 날짜로 채움
    inferred_ed.iloc[-1] = pd.to_datetime(date.today())
    # 실제 HST_ED_DT가 NaT인 곳에만 대입
    history_df['HST_ED_DT'] = history_df['HST_ED_DT'].fillna(inferred_ed)

    # 이제 모든 행에 유효한 종료일이 있음
    # 실제 근무 일수 계산 (종료일 포함)
    history_df['WORK_DAYS'] = (history_df['HST_ED_DT'] - history_df['HST_ST_DT']).dt.days + 1

    # 이력 구분별 합계 계산
    summary = history_df.groupby('HST_TYP')['WORK_DAYS'].sum().reset_index()
    summary.columns = ['HST_TYP', 'TOTAL_WORK_DAYS']

    print(f"[INFO] USER_ID '{user_id}' - 실제 근무 일수 요약:")
    print(summary)

    # CSV 저장
    summary.to_csv(f'actual_working_days_user_{user_id}.csv', index=False)
    print(f"[INFO] USER_ID '{user_id}' - 결과가 'actual_working_days_user_{user_id}.csv'에 저장되었습니다.")
def analyze_leave_return(user_id):
    """
    휴직 → 복직 이벤트를 페어링하여
    - 휴직 시작일(HST_ST_DT) → 복직 시작일(next HST_ST_DT where HST_TYP='복직') 사이 휴직 기간
    - 복직일 이후 다음 휴직 전까지의 근무 기간
    를 계산해 요약 및 CSV 저장
    """
    # 1) 히스토리 데이터 로드
    q = f"""
    SELECT HST_TYP, HST_ST_DT
    FROM ptbl_history
    WHERE USER_ID = '{user_id}'
      AND HST_TYP IN ('휴직', '복직')
    ORDER BY HST_ST_DT
    """
    df = load_data(q)
    if df.empty:
        print(f"[WARN] USER_ID '{user_id}'에 휴직/복직 이력이 없습니다.")
        return

    # 2) 날짜 변환
    df['HST_ST_DT'] = pd.to_datetime(df['HST_ST_DT'], errors='coerce')
    df = df.dropna(subset=['HST_ST_DT']).reset_index(drop=True)

    # 3) 휴직→복직 페어링
    pairs = []
    leave_date = None
    for idx, row in df.iterrows():
        if row['HST_TYP'] == '휴직':
            leave_date = row['HST_ST_DT']
        elif row['HST_TYP'] == '복직' and leave_date is not None:
            return_date = row['HST_ST_DT']
            leave_days = (return_date - leave_date).days
            pairs.append({
                'USER_ID': user_id,
                'LEAVE_DATE': leave_date.date(),
                'RETURN_DATE': return_date.date(),
                'LEAVE_DURATION_DAYS': leave_days
            })
            leave_date = None

    if not pairs:
        print(f"[WARN] USER_ID '{user_id}'에 짝을 이룬 휴직→복직 이벤트가 없습니다.")
        return

    result = pd.DataFrame(pairs)

    # 4) 출력 및 저장
    print(f"[INFO] USER_ID '{user_id}' - 휴직/복직 이벤트 분석:")
    print(result)
    result.to_csv(f'leave_return_summary_user_{user_id}.csv', index=False)
    print(f"[INFO] USER_ID '{user_id}' - 결과가 'leave_return_summary_user_{user_id}.csv'에 저장되었습니다.")

if __name__ == "__main__":
    try:
        # 사용자 목록 조회
        user_df = load_data("SELECT user_id FROM ptbl_user")
        if user_df.empty:
            print("[ERROR] ptbl_user 테이블이 비어 있습니다.")
            exit(1)

        user_ids = user_df['user_id'].tolist()
        sample_ids = random.sample(user_ids, min(10, len(user_ids)))

        for uid in sample_ids:
            print(f"\n=== USER_ID '{uid}' 작업 시작 ===")
            classify_work_history(uid)
            estimate_working_days(uid)
            analyze_leave_return(uid)

        print("\n프로젝트 이름: Work History Classification & Leave/Return Analysis")
    except Exception as e:
        print(f"[ERROR] 실행 중 예외 발생: {e}")