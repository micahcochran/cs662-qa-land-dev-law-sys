""" _summary_
"""

import sys

sys.path.append('../cs662-qa-land-dev-law-sys/')

from nlp.corpus import load_corpus

import os
import pandas as pd

file_path = f'{os.getcwd()}/programs/data'

print(file_path)

# TODO: Clean up this code for reuse
q_df = pd.read_csv(f'{file_path}/questions_input.csv', low_memory=False, names=['use', 'x', 'y', 'question'])
q_df['question'] = q_df['question'].str.replace('  ', ' ')
q_df['question'] = q_df['question'].str.replace(',', '')
q_df.drop_duplicates(subset=['question'], inplace=True, ignore_index=True)
q_df['document_identifier'] = 'x'
q_df['question_identifier'] = q_df.index
q_df['question_identifier'] = 'qid'+q_df['question_identifier'].astype(str)
print(q_df.shape)
q_dfs = []
q_dfs.append(q_df.iloc[:500])
q_dfs.append(q_df.iloc[500:1000])
q_dfs.append(q_df.iloc[1000:])

for index in range(13):
    for indx, q_df in enumerate(q_dfs):
        q_df['document_identifier'] = f'id{index}'
        q_df.to_csv(f'{file_path}/questions{index}{indx}.csv', index=False, header=True, lineterminator='\n', columns=['question', 'document_identifier', 'question_identifier']) 
    with open(f'{file_path}/text/text_{index}.txt', 'r') as file:
        data = file.read()
        data = data.replace('\n', ' ')
        data = data.replace('  ', ' ')
        data = data.rstrip()
        dataframe_dict = {'document_identifier': f'id{index}', 'document_text': data}
        d_df = pd.DataFrame.from_dict(dataframe_dict, orient='index').T
        d_df.to_csv(f'{file_path}/documents{index}.csv', index=False, header=True, lineterminator='\n', columns=['document_identifier', 'document_text'])
    with open(f'{file_path}/text/text_{index}.txt', 'w') as file:
        file.seek(0)
        file.write(data)
        file.truncate()

