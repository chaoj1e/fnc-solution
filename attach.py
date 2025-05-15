import pandas as pd

stances_path = "fnc-1-master/train_stances.random.csv"
bodies_path = "fnc-1-master/train_bodies.csv"

stances_df = pd.read_csv(stances_path)
bodies_df = pd.read_csv(bodies_path)

merged_df = pd.merge(stances_df, bodies_df, on="Body ID", how="left")

output_path = "fnc-1-master/train_merged.csv"
merged_df.to_csv(output_path, index=False)

print(f"Merged data saved to {output_path}, total {len(merged_df)} records.")
