"""
Semantic Parsing Phase - 2) Entity Linking and Class Linking
"""
# Python library imports
from collections import namedtuple

# External library imports
from loguru import logger
from nltk.metrics.distance import edit_distance
from nltk.tokenize import word_tokenize
from nltk.util import everygrams

# internal libraries
import indexes

class EntityClassLinking:
    def __init__(self):
        self.index_kg = indexes.IndexesKG()

    def ngram_collection(self, sentence):
        tokens = word_tokenize(sentence)
#        unigram = ngrams(tokens, 1)
#        bigram = ngrams(tokens, 2)
#        trigram = ngrams(tokens, 3)
        # compute unigram, bigram, and trigram at the same time
        # ngram_out = everygrams(tokens, max_len=3)  # this is per the paper
        # compute unigram, bigram, trigram, 4-grams and 5-grams all at the same time
        ngram_out = everygrams(tokens, max_len=5)  # However, I need more for my application

        # Do any of the indexes match these words?
#        indexes.IndexesKG()
        self.index_kg.label_index2
        self.index_kg.permitted_uses_index

    # this is Algorithm 1 in the paper
    def string_similarity_score(self, mention, candidates):
        # score a single mention and candidate
        def score_sts(m, cand):
            Score_Result = namedtuple("Score_STS", 'ld_div_ls, ld, ls, cand')
            ld = edit_distance(m, cand)
            ls = len(cand)
            return Score_Result(ld/ls, ld, ls, cand)

        computed_scores = [(score_sts(mention, cand)) for cand in candidates]
        # 2. Sort STS based on ld/ls
        computed_scores.sort(lambda x: x.ld_div_ls)
        # 3. Sort STS based on ls descending if ld is equal to zero
        if(any(filter(lambda x: x.ld == 0))):
            computed_scores.sort(lambda x: x.ls, reverse=True)

        # 4. Compute STR (STring Ranks) including entity candidates' string-ranks where the
        # string-rank of the entity candidate c is 1/index(stc) where index(stc) is the index
        # of the entity candicate c in STS Vector
        def compute_string_rank():
            pass
        # 5. Compute vector SES (SEmantic Similiariy) containing the cosine similarity cs
        # between the embeddings of an entity mention m and the embeddings of entity candidates
        def compute_semantic_similarity(self, mention):
            pass

        # 6. Sort SES based on cs descending

        # 7. Compute SER (SEmatic Ranks) including the entity candidates' semantic-ranks where
        # the semantic-rank of the entity candidate c is 1/index(sec) where index(sec) is the
        # index of the entity candidate c in SES vector
        def compute_semantic_ranks():
            pass

        # 8. Compute similarity score of an entity mention m and the entity candidate c based on
        # the sum of their ranks in STR and SER

    def run(self):
        sentence = 'Are auto-dismantling yards permitted?'
        ecl = EntityClassLinking()
        ecl.ngram_collection(sentence)
if __name__ == '__main__':
    pass