import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np

data_path = "fnc-1-master"
stances = pd.read_csv(os.path.join(data_path, "train_stances.csv"))
bodies = pd.read_csv(os.path.join(data_path, "train_bodies.csv"))
label_map = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}

merged = stances.merge(bodies, on='Body ID', how='left')
merged['label'] = merged['Stance'].map(label_map)
merged = merged.dropna(subset=['articleBody'])
merged['text'] = '[CLS] ' + merged['Headline'] + ' [SEP] ' + merged['articleBody'] + ' [SEP]'

train_texts, val_texts, train_labels, val_labels = train_test_split(
    merged['text'].tolist(), merged['label'].tolist(), test_size=0.2, stratify=merged['label'], random_state=42
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class FNC1Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FNC1Dataset(train_texts, train_labels, tokenizer)
val_dataset = FNC1Dataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
model.to(device)

class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2, 3]), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = criterion(outputs.logits, batch['labels'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true.extend(batch['labels'].cpu().numpy())

    acc = accuracy_score(true, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    print(classification_report(true, preds, target_names=label_map.keys()))

output_dir = "saved_model/bert_fnc1"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to: {output_dir}")
