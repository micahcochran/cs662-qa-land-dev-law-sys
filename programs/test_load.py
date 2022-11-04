import sys

sys.path.append('../cs662-qa-land-dev-law-sys/')

from nlp.corpus import load_corpus
from triplesdb import generate_template

corp = load_corpus("zo_corpus")
print(corp.raw_text[0])
print(f'Number of texts in {corp.name}: {len(corp.raw_text)}')
