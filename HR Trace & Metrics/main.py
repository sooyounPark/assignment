import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from loadData import load_data

# 1) 하이퍼파라미터
EMBED_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
MAX_LEN = 20  # 조직명 최대 토큰 길이

# 2) 간단 토크나이저 및 어휘 빌드
def build_vocab(texts, min_freq=2):
    from collections import Counter
    cnt = Counter()
    for t in texts:
        cnt.update(t.split())
    # 자주 나오는 단어만
    vocab = {w:i+2 for i,(w,f) in enumerate(cnt.items()) if f>=min_freq}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

def encode(text, vocab):
    tokens = text.split()
    idxs = [vocab.get(t, vocab['<UNK>']) for t in tokens][:MAX_LEN]
    # 패딩
    idxs += [vocab['<PAD>']] * (MAX_LEN - len(idxs))
    return idxs

# 3) Dataset 정의
class OrgDataset(Dataset):
    def __init__(self, df, vocab, label2idx):
        self.texts = df['ORG_FL_NM'].tolist()
        self.labels = [label2idx[y] for y in df['ORG_TYP']]
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        x = torch.tensor(encode(self.texts[i], self.vocab), dtype=torch.long)
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return x, y

# 4) 모델 정의
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        emb = self.embedding(x)                   # (B, L, E)
        _, (h_n, _) = self.lstm(emb)             # h_n: (1, B, H)
        out = self.fc(h_n.squeeze(0))            # (B, output_dim)
        return out

# 5) 학습 및 평가 함수
def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_model(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# 6) 전체 파이프라인
def classify_with_rnn():
    # 데이터 로드
    df_org = load_data("SELECT ORG_FL_NM, ORG_TYP FROM ptbl_org WHERE ORG_TYP IS NOT NULL")
    df_org = df_org[df_org['ORG_FL_NM'].notnull()].copy()
    # 어휘 및 레이블 매핑
    vocab = build_vocab(df_org['ORG_FL_NM'])
    labels = df_org['ORG_TYP'].unique().tolist()
    label2idx = {lab:i for i,lab in enumerate(labels)}

    # 학습/검증 분할
    train_df, val_df = train_test_split(df_org, test_size=0.2, random_state=42)

    # Dataset & DataLoader
    train_ds = OrgDataset(train_df, vocab, label2idx)
    val_ds   = OrgDataset(val_df,   vocab, label2idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # 모델, 옵티마이저, 손실함수
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(vocab_size=len(vocab), embed_dim=EMBED_DIM,
                           hidden_dim=HIDDEN_DIM, output_dim=len(labels)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 학습 루프
    for epoch in range(1, EPOCHS+1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc    = eval_model(model, val_loader, device)
        print(f"Epoch {epoch}/{EPOCHS} ▶ loss: {train_loss:.4f}, val_acc: {val_acc:.4f}")

    # 모델 저장
    torch.save(model.state_dict(), 'org_lstm_classifier.pt')
    print("✅ RNN 분류 모델 학습 완료 및 저장")

if __name__ == "__main__":
    classify_with_rnn()