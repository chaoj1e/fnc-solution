import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import numpy as np

MODEL_DIR = "saved_model/bert_fnc1"
INPUT_FILE = "related_train_data.csv"
OUTPUT_FILE = "bert_cls_vectors.npy"

df = pd.read_csv(INPUT_FILE)
texts = df["text"].tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
bert = BertModel.from_pretrained(MODEL_DIR)
bert.to(device)
bert.eval()

cls_vectors = []
with torch.no_grad():
    for text in tqdm(texts, desc="Extracting BERT [CLS] vectors"):
        encoding = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        encoding = {k: v.to(device) for k, v in encoding.items()}
        output = bert(**encoding)
        cls_vec = output.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        cls_vectors.append(cls_vec)

cls_array = torch.tensor(cls_vectors).numpy()
np.save(OUTPUT_FILE, cls_array)
print(f"BERT [CLS] vectors extracted and saved. Total: {len(cls_array)} → {OUTPUT_FILE}")
