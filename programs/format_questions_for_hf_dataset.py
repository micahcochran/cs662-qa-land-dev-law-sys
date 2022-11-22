""" _summary_
"""

import os
import pandas as pd
from datasets import load_dataset

file_path = f'{os.getcwd()}/programs/data'

# TODO: Clean up this code for reuse
hf_dataset_df = pd.read_csv(f'{file_path}/csv/questions_answers.csv', low_memory=False)

hf_dataset_df['question'] = hf_dataset_df['question'].str.replace('  ', ' ')
hf_dataset_df['question'] = hf_dataset_df['question'].str.replace(',', '')

hf_dataset_df.loc[hf_dataset_df['answer'] == 'True', 'answer'] = 'Yes'
hf_dataset_df.loc[hf_dataset_df['answer'] == 'False', 'answer'] = 'No' # ? may be a flaw in the dataset with no negative results it may not respond with a no

print(hf_dataset_df.head())

hf_dataset_df = hf_dataset_df.filter(['answer', 'question'], axis=1)

print(hf_dataset_df['question'].iloc[0])

hf_dataset_df.to_json(f'{file_path}/json/hf_QA_noSquad_dataset.json', orient='records')#, lines=True)

QA_dataset = load_dataset('json', data_files=f'{file_path}/json/hf_QA_noSquad_dataset.json')
# data_files = {"train": "QAZoningTrain.json", "test": "QAZoningTest.json"} # * this is how to load multiple files need to sklearn train_test_split into two sets first
# QA_dataset = load_dataset('json', data_files=data_files, split=['train', 'test'])
print(QA_dataset)