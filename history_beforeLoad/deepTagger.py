# deepTagger.py
import torch
import torch.nn as nn
from torchcrf import CRF
from konlpy.tag import Okt

okt = Okt()
START_TAG, STOP_TAG = "<START>", "<STOP>"

def tokenize_org_name(text):
    return [t[0] for t in okt.pos(text)]

class BiLSTMCRFTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=64, hidden_dim=128):
        super(BiLSTMCRFTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, x, tags=None, mask=None):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.fc(lstm_out)
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=mask)

# ✅ 추론용 간단한 예시 함수 (전라남도, 충청남도 등 앞단어 후보 중 예측)
def predict_prefix_tag(org_name):
    if org_name.startswith("전라남도"):
        return "전라남도"
    if "무안" in org_name or "완도" in org_name:
        return "전라남도"
    if "충남" in org_name or "천안" in org_name:
        return "충청남도"
    return "미상"