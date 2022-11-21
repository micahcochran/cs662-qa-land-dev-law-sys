""" _summary_
"""

import os
import sys
from pathlib import Path
from transformers import RobertaTokenizerFast

sys.path.append('../cs662-qa-land-dev-law-sys/')

from nlp.corpus import load_corpus
from nlp.model import create_custom_fast_tokenizer

corp = load_corpus("zo_corpus", tagset='universal_tagset')
cur_dir = os.getcwd()
model_dir = f'{cur_dir}/fast_tokenizers/{corp.name}'

Path(model_dir).mkdir(parents=True, exist_ok=True)

token = create_custom_fast_tokenizer(corp, model_dir, transform=RobertaTokenizerFast)

print(token.name)

token = RobertaTokenizerFast.from_pretrained(model_dir, max_len=512)

tokens = token.encode("This is a test")
print(tokens)
print(token.decode(tokens))