import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

STANCE_FILE = "fnc-1-master/competition_test_stances_unlabeled.csv"
BODY_FILE = "fnc-1-master/competition_test_bodies.csv"
MODEL_DIR = "saved_model/bert_fnc1"
OUTPUT_FILE = "prediction_result.csv"
BATCH_SIZE = 32
MAX_LEN = 512

assert os.path.exists(STANCE_FILE), "Test stance file not found"
assert os.path.exists(BODY_FILE), "Test body file not found"
assert os.path.exists(MODEL_DIR), "Model directory not found"

stances_df = pd.read_csv(STANCE_FILE)
bodies_df = pd.read_csv(BODY_FILE)
merged_df = stances_df.merge(bodies_df, on="Body ID", how="left")
merged_df["articleBody"] = merged_df["articleBody"].fillna("")
merged_df["text"] = "[CLS] " + merged_df["Headline"] + " [SEP] " + merged_df["articleBody"] + " [SEP]"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

class PredictDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}

    def __len__(self):
        return len(self.texts)

dataset = PredictDataset(merged_df["text"].tolist(), tokenizer)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

inv_label_map = {0: "agree", 1: "disagree", 2: "discuss", 3: "unrelated"}
all_preds = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Predicting"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())

result_df = merged_df[["Headline", "Body ID"]].copy()
result_df["Stance"] = [inv_label_map[p] for p in all_preds]
result_df.to_csv(OUTPUT_FILE, index=False)

print(f"Prediction completed. Results saved to: {OUTPUT_FILE}")
