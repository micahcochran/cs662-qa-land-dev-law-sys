""" _summary_
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset

if __name__ == "__main__": # needed for multiprocessing specifically load_dataset
    file_path = f'{os.getcwd()}/programs/data'

    # TODO: Clean up this code for reuse
    hf_dataset_df = pd.read_csv(f'{file_path}/csv/questions_answers.csv', low_memory=False)

    hf_dataset_df['question'] = hf_dataset_df['question'].str.replace('  ', ' ')
    hf_dataset_df['question'] = hf_dataset_df['question'].str.replace(',', '')

    hf_dataset_df.loc[hf_dataset_df['answer'] == 'True', 'answer'] = 'Yes'
    hf_dataset_df.loc[hf_dataset_df['answer'] == 'False', 'answer'] = 'No'
    
    print(hf_dataset_df.head())
    print(hf_dataset_df.shape)

    hf_dataset_df = hf_dataset_df.filter(['answer', 'question'], axis=1)
    
    uni = np.unique(hf_dataset_df['answer'], return_counts=True)
    print(uni)
    print(len(uni[0]))
    
    d = dict(enumerate(uni[0].flatten(), 1))
    inv_map = {v: k for k, v in d.items()}
    print(inv_map)
    
    hf_dataset_df = hf_dataset_df.replace({'answer': inv_map})
    
    hf_dataset_df.rename(columns={'answer': 'label', 'question': 'text'}, inplace=True)
    
    print(hf_dataset_df.head())

    # print(hf_dataset_df['question'].iloc[0])

    train, test = train_test_split(hf_dataset_df, random_state=246341428)

    train.to_json(f'{file_path}/json/QAZoningTrain.json', orient='records', lines=True)
    test.to_json(f'{file_path}/json/QAZoningTest.json', orient='records', lines=True)

    # Dataset creation example    
    data_files = {"train": f'{file_path}/json/QAZoningTrain.json', "test": f'{file_path}/json/QAZoningTest.json'} # * this is how to load multiple files, need to sklearn train_test_split into two sets first
    print(data_files)
    QA_dataset = load_dataset('json', data_files=data_files)
    print(QA_dataset)