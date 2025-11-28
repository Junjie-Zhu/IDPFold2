import os

import pandas as pd
import numpy as np

benchmark_files = [f for f in os.listdir('./data') if f.endswith('.csv')]
print('Found benchmark files:', benchmark_files)

df_list = []
for file in benchmark_files:
    df = pd.read_csv(f'./data/{file}')
    if 'length' not in df.columns:
        df['length'] = df['sequence'].apply(len)
    df_list.append(df)
_concat_df = pd.concat(df_list, ignore_index=True)
concat_df_len = len(_concat_df)

# first remove potential test_case duplicates
concat_df = _concat_df.drop_duplicates(subset=['test_case'])
print(f'Removed {concat_df_len - len(concat_df)} duplicate test_case entries.')
print('Removed entries:' + ', '.join(
    _concat_df[_concat_df.duplicated(subset=['test_case'], keep='first')]['test_case'].tolist()
))

# then group by sequence to combine test cases with same sequence
result_df = concat_df.groupby('sequence')['test_case'].apply(list)
result_df = result_df.reset_index(name='test_case')
print(f'Removed {len(concat_df) - len(result_df)} duplicate sequences.')

# update length column
result_df['length'] = result_df['sequence'].apply(len)
result_df['test_case'] = [result_df['test_case'][i] for i in range(len(result_df))]

print(f'Total unique sequences: {len(result_df)}')
print(f'Sequence with duplicate test cases: {np.sum(result_df["test_case"].apply(len) > 1)}')
print(f'Min Length: {np.min(result_df["length"])}, '
      f'Max Length: {np.max(result_df["length"])}, '
      f'Median Length: {np.median(result_df["length"])}')

# reformat test_case column to string for easier reading
result_df['test_case'] = result_df['test_case'].apply(lambda x: ':'.join(x))

result_df.to_csv('./all_benchmark.csv', index=False)
with open('./all_benchmark.fasta', 'w') as f:
    for i, row in result_df.iterrows():
        f.write(f'>{row["test_case"][0]}\n')
        f.write(f'{row["sequence"]}\n')
