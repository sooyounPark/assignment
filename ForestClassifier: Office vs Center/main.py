import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from loadData import load_data


def classify_work_history(user_id):
    ###############################
    # 1. 데이터베이스에서 데이터 불러오기
    ###############################
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

    ###############################
    # 2. 데이터 전처리
    ###############################
    # 조직 데이터 중 'ORG'와 'DEPT'로 분류된 데이터 선택
    org_df = org_df[org_df['ORG_TYP'].isin(['ORG', 'DEPT'])]
    if org_df.empty:
        print("[ERROR] 'ORG' 또는 'DEPT'로 분류된 조직 데이터가 없습니다.")
        return

    # Null 값 제거 및 공백 제거
    org_df['ORG_FL_NM'] = org_df['ORG_FL_NM'].fillna('').str.strip()
    org_df = org_df[org_df['ORG_FL_NM'] != '']
    if org_df.empty:
        print("[ERROR] ORG_FL_NM에 유효한 데이터가 없습니다.")
        return

    # TF-IDF 벡터화: 조직 이름의 텍스트 정보를 수치화
    vectorizer = TfidfVectorizer(stop_words=None)
    X = vectorizer.fit_transform(org_df['ORG_FL_NM'])
    y = org_df['ORG_TYP']

    ###############################
    # 3. 학습 및 테스트 데이터 분할
    ###############################
    # 전체 데이터를 80% 학습, 20% 테스트로 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ###############################
    # 4. 랜덤포레스트 모델 학습
    ###############################
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    ###############################
    # 5. 모델 평가 및 기본 시각화
    ###############################
    y_pred = clf.predict(X_test)
    print("[INFO] 모델 평가 결과:")
    print(classification_report(y_test, y_pred))

    # 시각화 1: 혼동 행렬 (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_user_{user_id}.png')
    plt.show()

    # 시각화 2: Feature Importance (상위 20개 특징 중요도)
    feature_names = vectorizer.get_feature_names_out()
    importances = clf.feature_importances_
    indices = importances.argsort()[::-1][:20]  # 상위 20개 특징
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette='viridis')
    plt.title('Top 20 Feature Importances in Random Forest Classifier')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'feature_importances_user_{user_id}.png')
    plt.show()

    # 시각화 3: ROC Curve 및 AUC (이진 분류에서 'DEPT'를 양성 클래스로 가정)
    # y_test에 대해 'DEPT'인 경우 1, 아니면 0인 이진 레이블 생성
    y_test_binary = (y_test == 'DEPT').astype(int)
    # 예측 확률: clf.predict_proba의 'DEPT' 클래스 인덱스 선택
    dept_index = list(clf.classes_).index('DEPT')
    y_score = clf.predict_proba(X_test)[:, dept_index]

    fpr, tpr, thresholds = roc_curve(y_test_binary, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'roc_curve_user_{user_id}.png')
    plt.show()

    # 시각화 4: t-SNE를 활용한 차원 축소 시각화
    # 먼저 PCA로 50차원으로 축소 후 t-SNE 적용 (고차원 TF-IDF 벡터)
    pca = PCA(n_components=50, random_state=42)
    X_train_pca = pca.fit_transform(X_train.toarray())
    tsne = TSNE(n_components=2, random_state=42)
    X_train_tsne = tsne.fit_transform(X_train_pca)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_train_tsne[:, 0], y=X_train_tsne[:, 1], hue=y_train, palette='Set1', alpha=0.7)
    plt.title('t-SNE Visualization of TF-IDF Features (Training set)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title='ORG_TYP')
    plt.tight_layout()
    plt.savefig(f'tsne_visualization_user_{user_id}.png')
    plt.show()

    ###############################
    # 6. 사용자 이력 데이터 예측
    ###############################
    # 사용자 이력 데이터 전처리: Null 제거 및 공백 제거
    history_df['HST_ORG_NM'] = history_df['HST_ORG_NM'].fillna('').str.strip()
    history_df = history_df[history_df['HST_ORG_NM'] != '']
    if history_df.empty:
        print("[ERROR] 유효한 조직명이 있는 이력 데이터가 없습니다.")
        return

    # TF-IDF 벡터화: 기존 vectorizer를 사용하여 변환
    new_X = vectorizer.transform(history_df['HST_ORG_NM'])
    predictions = clf.predict(new_X)

    ###############################
    # 7. 도메인 규칙 기반 보정
    ###############################
    corrected_predictions = []
    for org_name, prediction in zip(history_df['HST_ORG_NM'], predictions):
        if '소방서' in org_name and ('센터' not in org_name and '119안전센터' not in org_name):
            corrected_predictions.append('ORG')  # '소방서'만 포함된 경우 'ORG'로 보정
        elif '센터' in org_name or '119안전센터' in org_name:
            corrected_predictions.append('DEPT')  # '센터' 또는 '119안전센터'가 포함된 경우 'DEPT'로 보정
        else:
            corrected_predictions.append(prediction)

    # 결과 DataFrame 생성
    result_df = pd.DataFrame({
        'USER_ID': history_df['USER_ID'],
        'HST_ORG_NM': history_df['HST_ORG_NM'],
        'Predicted_ORG_TYP': predictions,
        'Corrected_ORG_TYP': corrected_predictions
    })

    print("[INFO] 사용자 이력 분류 결과:")
    print(result_df)

    # 시각화 5: 예측 분포 비교 (모델 예측 vs. 도메인 규칙 보정 후)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    result_df['Predicted_ORG_TYP'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Model Predicted Distribution')
    plt.xlabel('ORG_TYP')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    result_df['Corrected_ORG_TYP'].value_counts().plot(kind='bar', color='salmon')
    plt.title('Corrected Prediction Distribution')
    plt.xlabel('ORG_TYP')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'prediction_distribution_user_{user_id}.png')
    plt.show()

    ###############################
    # 8. 결과 CSV 파일 저장
    ###############################
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