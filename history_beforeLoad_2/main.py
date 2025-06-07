import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
import random
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
from loadData import load_data
from deepTagger import predict_prefix_tag
import platform


device = torch.device("cpu")
bert_model = BertModel.from_pretrained("bert-base-multilingual-cased").to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


def get_bert_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu()


class BERT_MLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=2):
        super(BERT_MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def rule_org_for_sobang_bonbu(name):
    # "소방본부"로 끝나면 ORG
    if re.search(r'소방.*본부$', name):
        return "ORG"
    # "소방서"로 끝나면 ORG
    elif re.search(r'소방서$', name):
        return "ORG"
    # "소방서 119" 등으로 이어지면 DEPT
    elif re.search(r'소방서\s+119', name):
        return "DEPT"
    return None
def preprocess_org_df(df):
    df = df[df['ORG_TYP'].isin(['ORG', 'DEPT'])].copy()
    df['ORG_FL_NM'] = df['ORG_FL_NM'].fillna('').str.strip()
    df.drop_duplicates(subset='ORG_FL_NM', inplace=True)
    df = df[df['ORG_FL_NM'] != '']

    suffix_terms = ['본부', '지원단', '센터', '부서', '소방서']

    def guess_prefix(name):
        for term in suffix_terms:
            if name == term or name.endswith(term):
                return f"전라남도 {name}" if not name.startswith("전라남도") else name
        return name

    df['ORG_FL_NM'] = df['ORG_FL_NM'].apply(guess_prefix)
    df['rule_guess'] = df['ORG_FL_NM'].apply(rule_org_for_sobang_bonbu)
    df['ORG_TYP'] = df.apply(lambda row: row['rule_guess'] if pd.notnull(row['rule_guess']) else row['ORG_TYP'], axis=1)
    df.drop(columns=['rule_guess'], inplace=True)

    return df


def visualize_results(df):
    if platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    elif platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:
        plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='user_id', y='accuracy', hue='user_id', palette='Blues_d', legend=False)
    plt.title('사용자별 정확도')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("user_accuracy_barplot.png")
    plt.close()

    class_f1_data = df[["dept_f1", "org_f1"]].mean().reset_index()
    class_f1_data.columns = ["class", "f1_score"]
    plt.figure(figsize=(8, 6))
    sns.barplot(data=class_f1_data, x="class", y="f1_score", palette="pastel")
    plt.title("클래스별 F1-score 평균")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("class_f1_score_comparison.png")
    plt.close()

    avg_f1_data = df[["macro_f1", "weighted_f1"]].mean().reset_index()
    avg_f1_data.columns = ["f1_type", "score"]
    plt.figure(figsize=(8, 6))
    sns.barplot(data=avg_f1_data, x="f1_type", y="score", palette="Set2")
    plt.title("Macro vs Weighted F1-score 평균")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("f1_score_comparison.png")
    plt.close()


def train_model():
    df = load_data("SELECT ORG_FL_NM, ORG_TYP FROM ptbl_org WHERE ORG_TYP IS NOT NULL")
    df = preprocess_org_df(df)

    print("학습 데이터 클래스 분포:")
    print(df['ORG_TYP'].value_counts())

    majority, minority = df['ORG_TYP'].value_counts().index
    df_majority = df[df['ORG_TYP'] == majority]
    df_minority = df[df['ORG_TYP'] == minority]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df = pd.concat([df_majority, df_minority_upsampled])

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['ORG_TYP'])
    texts = df['ORG_FL_NM'].tolist()
    emb = get_bert_embeddings(texts)

    # X_train, _, y_train, _ = train_test_split(emb, df['label'].values, test_size=0.2, stratify=df['label'], random_state=42)
    # X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    # y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_train = torch.tensor(emb, dtype=torch.float32).to(device)
    y_train = torch.tensor(df['label'].values, dtype=torch.long).to(device)

    model = BERT_MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return model, le


def evaluate_test_users(model, le, test_user_ids):
    all_results = []
    all_dfs = []

    for uid in test_user_ids:
        df = load_data("SELECT ORG_FL_NM, ORG_TYP FROM ptbl_org WHERE ORG_TYP IS NOT NULL")
        if df.empty:
            continue
        df = preprocess_org_df(df)
        df['label'] = le.transform(df['ORG_TYP'])

        texts = df['ORG_FL_NM'].tolist()
        emb = get_bert_embeddings(texts)
        X_tensor = emb.detach().clone().to(device)
        y_tensor = torch.tensor(df['label'].values, dtype=torch.long)

        with torch.no_grad():
            y_pred = model(X_tensor).argmax(dim=1).cpu().numpy()

        acc = accuracy_score(y_tensor.cpu().numpy(), y_pred)
        report = classification_report(y_tensor.cpu().numpy(), y_pred, target_names=le.classes_, output_dict=True)

        print(f"[{uid}] 평가 결과:")
        print(classification_report(y_tensor.cpu().numpy(), y_pred, target_names=le.classes_))

        df["predicted_prefix"] = df["ORG_FL_NM"].apply(predict_prefix_tag)
        df["ORG_FL_NM_보정"] = df.apply(lambda row: f"{row['predicted_prefix']} {row['ORG_FL_NM']}"
                                          if not row['ORG_FL_NM'].startswith(row['predicted_prefix']) else row['ORG_FL_NM'], axis=1)

        result_df = pd.DataFrame({
            "user_id": [uid] * len(df),
            "org_fl_nm(전처리 후)": df["ORG_FL_NM"],
            "org_fl_nm(보정 후)": df["ORG_FL_NM_보정"],
            "true_typ(실제 레이블)": le.inverse_transform(y_tensor.cpu().numpy()),
            "predicted_typ(예측 레이블)": le.inverse_transform(y_pred),
            "is_correct(예측 결과가 맞았는지 여부)": le.inverse_transform(y_tensor.cpu().numpy()) == le.inverse_transform(y_pred)
        })
        result_df.to_csv(f"user_result_detail_{uid}.csv", index=False, encoding="utf-8-sig")
        all_dfs.append(result_df)

        all_results.append({
            "user_id": uid,
            "accuracy": acc,
            "dept_f1": report.get("DEPT", {}).get("f1-score", 0),
            "org_f1": report.get("ORG", {}).get("f1-score", 0),
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"]
        })

    pd.concat(all_dfs).to_csv("all_user_result_detail.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(all_results)


def main():
    for f in glob.glob("*.png") + glob.glob("*.csv"):
        os.remove(f)

    user_df = load_data("SELECT user_id FROM ptbl_user")
    user_ids = user_df["user_id"].tolist()
    random.shuffle(user_ids)

    train_user_ids = user_ids[:1000]
    test_user_ids = random.sample(user_ids[1000:], min(10, len(user_ids) - 1000))

    print(f"\n▶ 학습 사용자 수: {len(train_user_ids)}명 / 테스트 사용자 수: {len(test_user_ids)}명")
    model, le = train_model()
    result_df = evaluate_test_users(model, le, test_user_ids)
    result_df.to_csv("bert_test_result.csv", index=False, encoding="utf-8-sig")
    visualize_results(result_df)
    print("\n✅ 전체 결과 저장 및 시각화 완료")


if __name__ == "__main__":
    main()
