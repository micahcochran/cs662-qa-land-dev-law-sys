""" _summary_
"""

import os
import sys
from pathlib import Path
from transformers import RobertaTokenizer
from tokenizers import ByteLevelBPETokenizer

sys.path.append('../cs662-qa-land-dev-law-sys/')

from nlp.corpus import load_corpus
from nlp.model import create_custom_tokenizer

corp = load_corpus("zo_corpus")

cur_dir = os.getcwd()
temp_dir = f'{cur_dir}/temp_data/{corp.name}'
model_dir = f'{cur_dir}/tokenizers/{corp.name}'

Path(temp_dir).mkdir(parents=True, exist_ok=True)
Path(model_dir).mkdir(parents=True, exist_ok=True)

tokenizer = ByteLevelBPETokenizer()

token = create_custom_tokenizer(corp, temp_dir, model_dir, tokenizer=tokenizer)

print(token.name)

token = RobertaTokenizer.from_pretrained(model_dir, max_len=512)

tokens = token.encode("This is a test")
print(tokens)
print(token.decode(tokens))

# from datasets import load_dataset

# bookcorpus = load_dataset("bookcorpus", split="train")

# print(type(bookcorpus[0]), '\n', bookcorpus[0])
