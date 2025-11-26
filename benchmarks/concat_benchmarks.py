import os

import pandas as pd
import numpy as np

benchmark_files = [f for f in os.listdir('./data') if f.endswith('.csv')]
df_list = []
for file in benchmark_files:
    df = pd.read_csv(f'./data/{file}')
    df_list.append(df)
concat_df = pd.concat(df_list, ignore_index=True)
result_df = concat_df.groupby('sequence')['test_case'].apply(list)
result_df = result_df.reset_index(name='test_case')

result_df['length'] = result_df['sequence'].apply(len)
result_df['test_case'] = [':'.join(result_df['test_case'][i]) for i in range(len(result_df))]

print(f'Total unique sequences: {len(result_df)}')
print(f'Sequence with duplicate test cases: {np.sum(result_df["test_case"].apply(len) > 1)}')
print(f'Min Length: {np.min(result_df["length"])}, Max Length: {np.max(result_df["length"])}')
print(f'Sequence of length < 300: {np.sum(result_df["length"] < 300)}')

result_df.to_csv('./all_benchmark.csv', index=False)

# save to fasta
with open('./all_benchmark.fasta', 'w') as f:
    for i, row in result_df.iterrows():
        f.write(f'>{row["test_case"]}\n')
        f.write(f'{row["sequence"]}\n')
