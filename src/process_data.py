import pandas as pd
import os

name = "chemdfm"

input_path = f"/HOME/paratera_xy/pxy547/HDD_POOL/for_verl/verl/data/molcap_{name}/train.parquet"
output_dir = f"/HOME/paratera_xy/pxy547/HDD_POOL/for_verl/verl/data/molcap_{name}/"
batch_size = 64*10

df = pd.read_parquet(input_path)

df = df.sample(frac=1, random_state=511).reset_index(drop=True)

num_batches = len(df) // batch_size

# 拆分并保存为多个 parquet 文件
for i in range(num_batches):
    batch_df = df.iloc[i * batch_size : (i + 1) * batch_size]
    output_path = os.path.join(output_dir, f"train_{i}.parquet")
    batch_df.to_parquet(output_path)

print(f"✅ Done: {num_batches} parquet files saved to {output_dir}")
