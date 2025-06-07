# 기존 모듈 + 추가 모듈
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import re
from sklearn.utils import resample

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from loadData import load_data
from konlpy.tag import Okt
from deepTagger import predict_prefix_tag  # 앞단어 추론 함수 호출

okt = Okt()


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform


def visualize_mlp_results(result_df):
    # 한글 폰트 설정
    if platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    elif platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:  # Linux
        plt.rcParams['font.family'] = 'NanumGothic'

    # 마이너스 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
    # [1] 사용자별 정확도
    plt.figure(figsize=(10, 6))
    sns.barplot(data=result_df, x='user_id', y='accuracy', palette='Blues_d')
    plt.title('사용자별 정확도 비교 (MLP)', fontsize=14)
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig("user_accuracy_barplot.png")
    plt.close()

    # [2] 클래스별 f1-score 비교 (ORG vs DEPT)
    plt.figure(figsize=(8, 6))
    class_f1_data = result_df[["org_f1", "dept_f1"]].mean().reset_index()
    class_f1_data.columns = ["class", "f1_score"]
    sns.barplot(data=class_f1_data, x="class", y="f1_score", palette="pastel")
    plt.title("클래스별 F1-score 평균 (ORG vs DEPT)", fontsize=14)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("class_f1_score_comparison.png")
    plt.close()

    # [3] macro vs weighted F1-score 비교
    plt.figure(figsize=(8, 6))
    avg_f1_data = result_df[["macro_f1", "weighted_f1"]].mean().reset_index()
    avg_f1_data.columns = ["f1_type", "score"]
    sns.barplot(data=avg_f1_data, x="f1_type", y="score", palette="Set2")
    plt.title("평균 F1-score 비교: Macro vs Weighted", fontsize=14)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("f1_score_comparison.png")
    plt.close()

def tokenize(text):
    return ['/'.join(t) for t in okt.pos(text)]

class CustomTfidfVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        return lambda doc: tokenize(doc)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def rule_org_for_sobang_bonbu(org_name):
    if re.search(r'소방.*본부$', org_name):
        return "ORG"
    return None

def apply_manual_rules(df):
    df['ORG_TYP_RULE'] = df['ORG_TYP']
    df['ORG_FL_NM'] = df['ORG_FL_NM'].str.strip()
    for idx, row in df.iterrows():
        org_name = row['ORG_FL_NM']
        rule_typ = rule_org_for_sobang_bonbu(org_name)
        if rule_typ:
            df.at[idx, 'ORG_TYP_RULE'] = rule_typ
            continue
        if '본부' in org_name and org_name.endswith('본부'):
            df.at[idx, 'ORG_TYP_RULE'] = 'ORG'
            continue
        if '본부' in org_name and not org_name.endswith('본부'):
            df.at[idx, 'ORG_TYP_RULE'] = 'ORG'
            continue
        if any(term in org_name for term in ['지원단', '센터']):
            df.at[idx, 'ORG_TYP_RULE'] = 'DEPT'
            continue
        if org_name.endswith('소방서'):
            df.at[idx, 'ORG_TYP_RULE'] = 'ORG'
            continue
    df['ORG_TYP'] = df['ORG_TYP_RULE']
    return df

def preprocess_org_df(df):
    df = df[df['ORG_TYP'].isin(['ORG', 'DEPT'])].copy()
    df['ORG_FL_NM_RAW'] = df['ORG_FL_NM']
    df['ORG_FL_NM'] = df['ORG_FL_NM'].fillna('').str.strip()
    df = df[df['ORG_FL_NM'] != '']

    suffix_terms = ['본부', '지원단', '센터', '부서', '소방서', '119안전센터', '119지역대']
    def guess_prefix(name):
        for term in suffix_terms:
            if name == term or name.endswith(term):
                return f"전라남도 {name}" if not name.startswith("전라남도") else name
        return name

    df['ORG_FL_NM'] = df['ORG_FL_NM'].apply(guess_prefix)
    return df

def train_classifier():
    org_df = load_data("SELECT ORG_FL_NM, ORG_TYP FROM ptbl_org WHERE ORG_TYP IS NOT NULL")
    org_df = preprocess_org_df(org_df)
    org_df = apply_manual_rules(org_df)

    # 🔁 Oversampling (불균형 수정)
    majority_class = org_df['ORG_TYP'].value_counts().idxmax()
    minority_class = org_df['ORG_TYP'].value_counts().idxmin()

    df_majority = org_df[org_df['ORG_TYP'] == majority_class]
    df_minority = org_df[org_df['ORG_TYP'] == minority_class]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    org_df = pd.concat([df_majority, df_minority_upsampled])

    vectorizer = CustomTfidfVectorizer()
    X = vectorizer.fit_transform(org_df['ORG_FL_NM'])

    le = LabelEncoder()
    y = le.fit_transform(org_df['ORG_TYP'])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    model = MLPClassifier(input_dim=X.shape[1], output_dim=len(le.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "trained_mlp.pth")
    print(org_df['ORG_TYP'].value_counts())  # 클래스 균형 확인용
    return model, vectorizer, le

def merge_with_predicted_prefix(row):
    org = row["org_fl_nm(전처리 후)"]
    prefix = row["predicted_prefix"]

    # 이미 prefix가 org의 앞에 붙어있는 경우는 그대로 사용
    if org.startswith(prefix):
        return org
    elif prefix in org:
        return org  # 중간 포함된 경우도 중복 제거
    else:
        return f"{prefix} {org}"
def test_classifier(model, vectorizer, le, user_id):
    org_df = load_data("SELECT ORG_FL_NM, ORG_TYP FROM ptbl_org WHERE ORG_TYP IS NOT NULL")
    org_df = preprocess_org_df(org_df)
    org_df = apply_manual_rules(org_df)

    X = vectorizer.transform(org_df['ORG_FL_NM'])
    y = le.transform(org_df['ORG_TYP'])
    X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_tensor)
        y_pred = torch.argmax(y_pred_tensor, dim=1).numpy()

    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    print(f"[{user_id}] 평가 결과:")
    print(classification_report(y, y_pred, target_names=le.classes_))

    results_df = pd.DataFrame({
        "user_id": [user_id] * len(org_df),
        "org_fl_nm_raw(원본)": org_df["ORG_FL_NM_RAW"],
        "org_fl_nm(전처리 후)": org_df["ORG_FL_NM"],
        "true_typ(실제 레이블)": le.inverse_transform(y),
        "predicted_typ(예측 레이블)": le.inverse_transform(y_pred),
        "is_correct(예측 결과가 맞았는지 여부)": le.inverse_transform(y) == le.inverse_transform(y_pred)
    })

    # ➕ 앞단어 추론 적용
    results_df["predicted_prefix"] = results_df["org_fl_nm(전처리 후)"].apply(predict_prefix_tag)
    results_df["org_fl_nm(앞단어추론 보정)"] = results_df.apply(merge_with_predicted_prefix, axis=1)

    results_df.to_csv(f"user_result_detail_{user_id}.csv", index=False, encoding="utf-8-sig")

    return {
        "user_id": user_id,
        "accuracy": acc,
        "dept_precision": report["DEPT"]["precision"],
        "dept_recall": report["DEPT"]["recall"],
        "dept_f1": report["DEPT"]["f1-score"],
        "org_precision": report["ORG"]["precision"],
        "org_recall": report["ORG"]["recall"],
        "org_f1": report["ORG"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"]
    }

def main():
    try:
        for file in glob.glob("*.png"):
            os.remove(file)
        for file in glob.glob("*.csv"):
            os.remove(file)

        user_df = load_data("SELECT user_id FROM ptbl_user")
        user_ids = user_df["user_id"].tolist()
        random.shuffle(user_ids)

        train_user_ids = user_ids[:1000]
        test_user_ids = random.sample(user_ids[1000:], min(10, len(user_ids) - 1000))

        print(f"학습용 사용자 수: {len(train_user_ids)}명")
        print(f"테스트용 사용자 수: {len(test_user_ids)}명")

        print("\n[STEP 1] 모델 학습 중...")
        model, vectorizer, le = train_classifier()

        print("\n[STEP 2] 테스트 사용자 10명에 대해 성능 평가:")
        all_results = []
        all_dfs = []

        for uid in test_user_ids:
            result = test_classifier(model, vectorizer, le, uid)
            all_results.append(result)
            df = pd.read_csv(f"user_result_detail_{uid}.csv", encoding='utf-8-sig')
            all_dfs.append(df)

        result_df = pd.DataFrame(all_results)
        result_df.to_csv("mlp_test_result.csv", index=False, encoding="utf-8-sig")
        all_result_df = pd.concat(all_dfs, ignore_index=True)
        all_result_df.to_csv("all_user_result_detail.csv", index=False, encoding="utf-8-sig")

        visualize_mlp_results(result_df)

        print("\n✅ 결과 저장 완료.")
    except Exception as e:
        print(f"[ERROR] 실행 중 예외 발생: {e}")

if __name__ == "__main__":
    main()