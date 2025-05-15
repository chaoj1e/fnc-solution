import os
import pandas as pd

data_path = "fnc-1-master"
stance_file = os.path.join(data_path, "train_stances.csv")
body_file = os.path.join(data_path, "train_bodies.csv")

stances = pd.read_csv(stance_file)
bodies = pd.read_csv(body_file)

merged = stances.merge(bodies, on='Body ID', how='left')
label_map = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
merged['label'] = merged['Stance'].map(label_map)
merged = merged.dropna(subset=['articleBody'])

merged['text'] = '[CLS] ' + merged['Headline'] + ' [SEP] ' + merged['articleBody'] + ' [SEP]'

related_df = merged[merged['label'] != 3].copy()
related_df = related_df[['Headline', 'articleBody', 'label', 'text']]

related_df.to_csv("related_train_data.csv", index=False)
print(f"Related samples saved to related_train_data.csv, total {len(related_df)} records.")
