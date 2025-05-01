import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 비인터랙티브 백엔드를 사용하여 HTTP 429 에러 회피
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from loadData import load_data
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False  # 음수 기호가 깨지지 않도록 설정
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
    # 조직 데이터 중 'ORG'와 'DEPT'만 선택 (copy()로 명시적 복사)
    org_df = org_df[org_df['ORG_TYP'].isin(['ORG', 'DEPT'])].copy()
    if org_df.empty:
        print("[ERROR] 'ORG' 또는 'DEPT'로 분류된 조직 데이터가 없습니다.")
        return

    # Null 값 처리 및 앞뒤 공백 제거 후, copy() 적용
    org_df['ORG_FL_NM'] = org_df['ORG_FL_NM'].fillna('').str.strip()
    org_df = org_df[org_df['ORG_FL_NM'] != ''].copy()
    if org_df.empty:
        print("[ERROR] ORG_FL_NM에 유효한 데이터가 없습니다.")
        return

    # TF-IDF 벡터화: 조직 이름 텍스트를 수치형 피처로 변환
    vectorizer = TfidfVectorizer(stop_words=None)
    X = vectorizer.fit_transform(org_df['ORG_FL_NM'])
    y = org_df['ORG_TYP']

    ##############################
    # 3. 학습/테스트 데이터 분할
    ##############################
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # (a) 혼동 행렬 시각화: Figure 1
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Figure 1. Random Forest Confusion Matrix (USER_ID: {user_id})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_user_{user_id}.png')
    plt.close()

    # (b) 특징 중요도 시각화: Figure 2
    feature_names = vectorizer.get_feature_names_out()
    importances = clf.feature_importances_
    indices = importances.argsort()[::-1][:20]  # 상위 20개 특징 선택
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette='viridis')
    plt.title(f'Figure 2. Top 20 Feature Importances (USER_ID: {user_id})')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'feature_importances_user_{user_id}.png')
    plt.close()

    # (c) ROC Curve 및 AUC 시각화: Figure 3
    # 'DEPT'를 양성 클래스로 가정하여 이진 분류 평가
    y_test_binary = (y_test == 'DEPT').astype(int)
    dept_index = list(clf.classes_).index('DEPT')
    y_score = clf.predict_proba(X_test)[:, dept_index]

    fpr, tpr, thresholds = roc_curve(y_test_binary, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Figure 3. ROC Curve (USER_ID: {user_id})')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'roc_curve_user_{user_id}.png')
    plt.close()

    # (d) t-SNE 시각화: Figure 4
    # 고차원 TF-IDF 피처를 PCA로 50차원 축소 후, t-SNE를 활용해 2차원 시각화
    pca = PCA(n_components=50, random_state=42)
    X_train_pca = pca.fit_transform(X_train.toarray())
    tsne = TSNE(n_components=2, random_state=42)
    X_train_tsne = tsne.fit_transform(X_train_pca)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_train_tsne[:, 0], y=X_train_tsne[:, 1], hue=y_train, palette='Set1', alpha=0.7)
    plt.title(f'Figure 4. t-SNE Visualization (USER_ID: {user_id})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title='ORG_TYP')
    plt.tight_layout()
    plt.savefig(f'tsne_visualization_user_{user_id}.png')
    plt.close()

    # (e) 예측 분포 비교 시각화: Figure 5
    # 사용자 이력 데이터 전처리 및 TF-IDF 벡터화 후 예측 (copy() 적용)
    history_df['HST_ORG_NM'] = history_df['HST_ORG_NM'].fillna('').str.strip()
    history_df = history_df[history_df['HST_ORG_NM'] != ''].copy()
    if history_df.empty:
        print("[ERROR] 유효한 조직명이 있는 이력 데이터가 없습니다.")
        return

    all_org_names = org_df['ORG_FL_NM'].tolist()
    vectorizer_refine = TfidfVectorizer().fit(all_org_names + history_df['HST_ORG_NM'].tolist())
    org_vec = vectorizer_refine.transform(all_org_names)
    history_vec = vectorizer_refine.transform(history_df['HST_ORG_NM'])

    original_names = history_df['HST_ORG_NM'].tolist()
    refined_names = []
    for i, vec in enumerate(history_vec):
        sim_scores = cosine_similarity(vec, org_vec).flatten()
        max_score = sim_scores.max()
        if max_score > 0.7:
            matched_index = sim_scores.argmax()
            refined_names.append(all_org_names[matched_index])
        else:
            refined_names.append(original_names[i])

    history_df['Original_HST_ORG_NM'] = original_names
    history_df['Refined_HST_ORG_NM'] = refined_names
    history_df['HST_ORG_NM'] = refined_names  # 예측에 사용될 최종 컬럼


    # (e) 사용자 이력 데이터 전처리 후, TF-IDF 벡터화 및 유사 조직명 보정
    history_df['HST_ORG_NM'] = history_df['HST_ORG_NM'].fillna('').str.strip()
    history_df = history_df[history_df['HST_ORG_NM'] != ''].copy()
    if history_df.empty:
        print("[ERROR] 유효한 조직명이 있는 이력 데이터가 없습니다.")
        return

    # 유사도 기반 보정을 위한 TF-IDF 벡터화
    all_org_names = org_df['ORG_FL_NM'].tolist()
    vectorizer_refine = TfidfVectorizer().fit(all_org_names + history_df['HST_ORG_NM'].tolist())
    org_vec = vectorizer_refine.transform(all_org_names)
    history_vec = vectorizer_refine.transform(history_df['HST_ORG_NM'])

    # 유사 조직명 치환 (임계값 0.7 이상)
    refined_names = []
    for i, vec in enumerate(history_vec):
        sim_scores = cosine_similarity(vec, org_vec).flatten()
        max_score = sim_scores.max()
        if max_score > 0.7:
            matched_index = sim_scores.argmax()
            refined_names.append(all_org_names[matched_index])
        else:
            refined_names.append(history_df.iloc[i]['HST_ORG_NM'])

    history_df['Original_HST_ORG_NM'] = history_df['HST_ORG_NM']
    history_df['Refined_HST_ORG_NM'] = refined_names
    history_df['HST_ORG_NM'] = refined_names  # 예측에 사용될 컬럼 덮어쓰기


    new_X = vectorizer.transform(history_df['HST_ORG_NM'])
    predictions = clf.predict(new_X)

    # 도메인 규칙 기반 보정
    corrected_predictions = []
    for org_name, prediction in zip(history_df['HST_ORG_NM'], predictions):
        if '소방서' in org_name and ('센터' not in org_name and '119안전센터' not in org_name):
            corrected_predictions.append('ORG')
        elif '센터' in org_name or '119안전센터' in org_name:
            corrected_predictions.append('DEPT')
        else:
            corrected_predictions.append(prediction)

    result_df = pd.DataFrame({
        'USER_ID': history_df['USER_ID'],
        'Original_HST_ORG_NM': history_df['Original_HST_ORG_NM'],
        'Refined_HST_ORG_NM': history_df['Refined_HST_ORG_NM'],
        'Predicted_ORG_TYP': predictions,
        'Corrected_ORG_TYP': corrected_predictions
    })

    print(f"[INFO] USER_ID '{user_id}' - 사용자 이력 분류 결과:")
    print(result_df)

    # 예측 분포 비교 (보정 전 vs. 보정 후)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    result_df['Predicted_ORG_TYP'].value_counts().plot(kind='bar', color='skyblue')
    plt.title(f'Figure 5. Model Predicted Distribution (USER_ID: {user_id})')
    plt.xlabel('ORG_TYP')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    result_df['Corrected_ORG_TYP'].value_counts().plot(kind='bar', color='salmon')
    plt.title(f'Figure 5. Corrected Prediction Distribution (USER_ID: {user_id})')
    plt.xlabel('ORG_TYP')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(f'prediction_distribution_user_{user_id}.png')
    plt.close()

    ##############################
    # 6. 결과 CSV 파일 저장
    ##############################
    output_file = f'classified_work_history_user_{user_id}.csv'
    result_df.to_csv(output_file, index=False)
    print(f"[INFO] USER_ID '{user_id}' - 분류 결과가 '{output_file}'에 저장되었습니다.")


if __name__ == "__main__":
    try:
        ##############################
        # 사용자 목록에서 랜덤으로 10명 선정하기
        ##############################
        user_query = "SELECT user_id FROM ptbl_user"
        user_df = load_data(user_query)
        if user_df.empty:
            print("[ERROR] 사용자 목록(ptbl_user)이 비어있습니다.")
            exit(1)

        # 사용자 목록에서 user_id를 리스트로 추출 후, 무작위로 10명 선택
        user_ids = user_df["user_id"].tolist()
        sample_user_ids = random.sample(user_ids, min(10, len(user_ids)))

        # 각 user_id마다 분류 작업 수행
        for uid in sample_user_ids:
            print(f"\n========== USER_ID '{uid}'에 대한 분류 작업을 시작합니다 ==========")
            classify_work_history(uid)

        print("프로젝트 이름: Work History Classification Project")
    except Exception as e:
        print(f"[ERROR] 실행 중 오류 발생: {e}")