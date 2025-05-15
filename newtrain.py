import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("related_train_data.csv")
bert_vectors = np.load("bert_cls_vectors.npy").astype(np.float32) * 0.2

sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sent_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sent_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name).to(device)
sent_model.eval()

agree_keywords = ['agree', 'support', 'confirm', 'true', 'verified']
disagree_keywords = ['disagree', 'oppose', 'deny', 'false', 'fake', 'refute']
stance_hint_set = ["agree", "disagree", "conflict", "neutral"]

def get_sentiment_score(text):
    inputs = sent_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = sent_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    return scores[0][2].item()

def get_stance_hint(headline, body):
    h, b = headline.lower(), body.lower()
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

sentiment_diffs, stance_hints = [], []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting rule features"):
    s1 = get_sentiment_score(row["Headline"])
    s2 = get_sentiment_score(row["articleBody"])
    sentiment_diffs.append(abs(s1 - s2))
    stance_hints.append(get_stance_hint(row["Headline"], row["articleBody"]))

onehot = np.zeros((len(stance_hints), len(stance_hint_set)))
for i, hint in enumerate(stance_hints):
    if hint in stance_hint_set:
        onehot[i][stance_hint_set.index(hint)] = 1

extra_features = np.hstack([
    np.array(sentiment_diffs).reshape(-1, 1).astype(np.float32),
    onehot.astype(np.float32)
])
X = np.hstack([bert_vectors, extra_features])
y = df["label"].values.astype(np.int64)

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train).to(device), torch.tensor(y_train).to(device)),
    batch_size=64, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val).to(device), torch.tensor(y_val).to(device)),
    batch_size=64
)

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

model = FusionMLP(X.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Training Loss: {total_loss:.4f}")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

print("\nFusion model classification report:")
print(classification_report(all_labels, all_preds, target_names=["agree", "disagree", "discuss"]))
print(f"Validation accuracy: {accuracy_score(all_labels, all_preds):.4f}")

os.makedirs("saved_model", exist_ok=True)
torch.save(model.state_dict(), "saved_model/fusion_mlp_torch.pth")
print("Model saved to saved_model/fusion_mlp_torch.pth")
