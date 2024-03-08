import glob
import pickle
import torch
import pandas as pd
import fsspec
from baselines.apply_filter import load_metadata
import os

file_list = glob.glob('/local1/siting/scores/*.pt')
df_original = load_metadata('/local1/datasets/datacomp_small/metadata/', num_workers=os.cpu_count())
for file_path in file_list:
    zipped_content = torch.load(file_path)
    meru_uid_collection, meru_score_collection = zip(*zipped_content)
    df = pd.DataFrame({'uid': meru_uid_collection, 'l_xtime': meru_score_collection})
    new_df = df_original[df_original['uid'].isin(df['uid'])]
    merged_df = pd.merge(new_df, df, on='uid')

    merged_df.to_parquet('/local1/siting/metadata/'+file_path[-15:-7]+'.parquet')