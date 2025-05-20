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
from sklearn.metrics.pairwise import cosine_similarity
from loadData import load_data

from konlpy.tag import Okt
okt = Okt()

def tokenize(text):
    return ['/'.join(t) for t in okt.pos(text)]

class CustomTfidfVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        return lambda doc: tokenize(doc)

import matplotlib.pyplot as plt
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False  # 음수 기호가 깨지지 않도록 설정

def classify_work_history(user_id):
    ##############################
    # 1. 데이터 불러오기
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
    org_df['ORG_FL_NM'] = org_df['ORG_FL_NM'].fillna('').str.strip()
    org_df = org_df[org_df['ORG_FL_NM'] != ''].copy()
    if org_df.empty:
        print("[ERROR] ORG_FL_NM에 유효한 데이터가 없습니다.")
        return

    # 형태소 기반 TF-IDF 벡터화
    vectorizer = CustomTfidfVectorizer()
    X = vectorizer.fit_transform(org_df['ORG_FL_NM'])
    y = org_df['ORG_TYP']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"[INFO] USER_ID '{user_id}' - 모델 평가 결과:")
    print(classification_report(y_test, y_pred))

    # 이하 시각화 및 보정 로직은 그대로 유지하되, vectorizer 부분에 CustomTfidfVectorizer 사용
    # ... (기존 시각화, 예측 분포, tsne 등 동일)

    # 조직명 유사도 기반 정제
    history_df['HST_ORG_NM'] = history_df['HST_ORG_NM'].fillna('').str.strip()
    history_df = history_df[history_df['HST_ORG_NM'] != ''].copy()

    all_org_names = org_df['ORG_FL_NM'].tolist()
    vectorizer_refine = CustomTfidfVectorizer().fit(all_org_names + history_df['HST_ORG_NM'].tolist())
    org_vec = vectorizer_refine.transform(all_org_names)
    history_vec = vectorizer_refine.transform(history_df['HST_ORG_NM'])

    refined_names = []
    for i, vec in enumerate(history_vec):
        sim_scores = cosine_similarity(vec, org_vec).flatten()
        max_score = sim_scores.max()
        original = history_df.iloc[i]['HST_ORG_NM']
        if max_score > 0.7:
            matched = all_org_names[sim_scores.argmax()]
            if matched != original and (matched.startswith(original) or len(matched) > len(original)):
                refined_names.append(matched)
            else:
                refined_names.append(original)
        else:
            refined_names.append(original)

    history_df['Original_HST_ORG_NM'] = history_df['HST_ORG_NM']
    history_df['Refined_HST_ORG_NM'] = refined_names
    history_df['HST_ORG_NM'] = refined_names

    new_X = vectorizer.transform(history_df['HST_ORG_NM'])
    predictions = clf.predict(new_X)

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

    output_file = f'classified_work_history_user_{user_id}.csv'
    result_df.to_csv(output_file, index=False)
    print(f"[INFO] USER_ID '{user_id}' - 분류 결과가 '{output_file}'에 저장되었습니다.")

if __name__ == "__main__":
    try:
        user_query = "SELECT user_id FROM ptbl_user"
        user_df = load_data(user_query)
        if user_df.empty:
            print("[ERROR] 사용자 목록(ptbl_user)이 비어있습니다.")
            exit(1)

        user_ids = user_df["user_id"].tolist()
        sample_user_ids = random.sample(user_ids, min(10, len(user_ids)))

        for uid in sample_user_ids:
            print(f"\n========== USER_ID '{uid}'에 대한 분류 작업을 시작합니다 ==========")
            classify_work_history(uid)

        print("프로젝트 이름: Work History Classification Project")
    except Exception as e:
        print(f"[ERROR] 실행 중 오류 발생: {e}")
