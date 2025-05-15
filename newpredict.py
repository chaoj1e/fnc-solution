import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ==== 配置 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERT_MODEL_DIR = "saved_model/bert_fnc1"
MLP_MODEL_PATH = "saved_model/fusion_mlp_torch.pth"
STANCE_PATH = "fnc-1-master/competition_test_stances_unlabeled.csv"
BODY_PATH = "fnc-1-master/competition_test_bodies.csv"
OUTPUT_PATH = "fusion_prediction_result.csv"

# ==== 加载测试数据 ====
stances_df = pd.read_csv(STANCE_PATH)
bodies_df = pd.read_csv(BODY_PATH)
merged_df = stances_df.merge(bodies_df, on="Body ID", how="left")
merged_df["articleBody"] = merged_df["articleBody"].fillna("")
merged_df["text"] = "[CLS] " + merged_df["Headline"] + " [SEP] " + merged_df["articleBody"] + " [SEP]"

# ==== BERT模型 ====
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)
bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR, output_hidden_states=True).to(device)
bert_model.eval()

# ==== 情感模型 ====
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sent_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sent_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name).to(device)
sent_model.eval()

# ==== 定义关键词辅助特征 ====
agree_keywords = ['agree', 'support', 'confirm', 'true', 'verified']
disagree_keywords = ['disagree', 'oppose', 'deny', 'false', 'fake', 'refute']
stance_hint_set = ["agree", "disagree", "conflict", "neutral"]

def get_sentiment_score(text):
    inputs = sent_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = sent_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs[0][2].item()

def get_stance_hint(headline, body):
    h = headline.lower()
    b = body.lower()
    agree = any(kw in h or kw in b for kw in agree_keywords)
    disagree = any(kw in h or kw in b for kw in disagree_keywords)
    if agree and not disagree:
        return "agree"
    elif disagree and not agree:
        return "disagree"
    elif agree and disagree:
        return "conflict"
    else:
        return "neutral"

# ==== 定义MLP模型结构 ====
class FusionMLP(nn.Module):
    def __init__(self, input_dim=773):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.net(x)

mlp_model = FusionMLP(input_dim=773).to(device)
mlp_model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=device))
mlp_model.eval()

# ==== 提取CLS向量并做BERT初筛 ====
texts = merged_df["text"].tolist()
cls_vectors = []
bert_preds = []

batch_size = 32
for i in tqdm(range(0, len(texts), batch_size), desc="BERT提取中"):
    batch_texts = texts[i:i+batch_size]
    encoded = bert_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**encoded)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        bert_preds.extend(pred.tolist())
        cls = outputs.hidden_states[-1][:, 0, :].detach().cpu().numpy()
        cls_vectors.append(cls)

cls_vectors = np.vstack(cls_vectors) * 0.2  # BERT缩放权重
merged_df["bert_pred"] = bert_preds

# ==== 融合 MLP 推理 ====
final_preds = []

for i, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="融合判断中"):
    if row["bert_pred"] == 3:
        final_preds.append("unrelated")
    else:
        s1 = get_sentiment_score(row["Headline"])
        s2 = get_sentiment_score(row["articleBody"])
        s_diff = abs(s1 - s2)

        stance_hint = get_stance_hint(row["Headline"], row["articleBody"])
        onehot = np.zeros(len(stance_hint_set), dtype=np.float32)
        onehot[stance_hint_set.index(stance_hint)] = 1

        feat = np.hstack([cls_vectors[i], [s_diff], onehot]).astype(np.float32)
        with torch.no_grad():
            pred_tensor = torch.tensor(feat).unsqueeze(0).to(device)
            output = mlp_model(pred_tensor)
            label = torch.argmax(output, dim=1).item()
            final_preds.append(["agree", "disagree", "discuss"][label])

# ==== 保存CSV ====
output_df = merged_df[["Headline", "Body ID"]].copy()
output_df["Stance"] = final_preds
output_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ 融合预测完成，已保存至：{OUTPUT_PATH}")
