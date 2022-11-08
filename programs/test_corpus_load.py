""" _summary_
"""

from nlp.corpus import load_corpus

import nltk
from nltk.corpus import brown
from triplesdb import generate_template
nltk.download('brown')

corp = load_corpus("zo_corpus")
print(f'Number of texts in {corp.name}: {len(corp.raw_texts)}')
print(f'Number of sentences in {corp.name}: {len(corp.sents)}')
print(f'Number of words in {corp.name}: {len(corp.words)}')
print(f"Vocabulary size: {len(corp.vocabulary)}")
print(f"Number of tagged sentences in {corp.name}: {len(corp.tagged_sents)}")
print(f"Number of tagged words in {corp.name}: {len(corp.tagged_words)}")

# print(corp.raw_texts[0])
