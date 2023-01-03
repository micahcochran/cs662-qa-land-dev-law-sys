"""
Semantic Parsing Phase - 2) Entity Linking and Class Linking
"""
# Python library imports
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import List, Optional, Tuple

# External library imports
from nltk.metrics.distance import edit_distance
from nltk.tokenize import word_tokenize
from nltk.util import everygrams
from sentence_transformers import SentenceTransformer, util

# internal libraries
import indexes

class EntityClassLinking:
    def __init__(self, index_kg: Optional[indexes.IndexesKG] = None, verbose=True):
        if index_kg:
            self.index_kg = index_kg
        else:
            self.index_kg = indexes.IndexesKG()

        # SBERT for computing the embeddings
        # this model is a 5 times faster model
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        # This is for step 5 Algorithm 1.
        ec_embeddings = self.sbert_model.encode(self.index_kg.all_entity_labels(), convert_to_tensor=True)
        self.entity_candidate_embeddings = list(zip(self.index_kg.all_entity_labels(), ec_embeddings))

        self.verbose = verbose

    def ngram_collection(self, sentence) -> list:
        tokens = word_tokenize(sentence)
        # compute unigram, bigram, and trigram at the same time
        # ngram_out = everygrams(tokens, max_len=3)  # this is per the paper
        # compute unigram, bigram, trigram, 4-grams and 5-grams all at the same time
        ngram_out = list(everygrams(tokens, max_len=5))  # However, I need more for my application.
        # print(ngram_out)


        return ngram_out

    # this is Algorithm 1 in the paper
    #
    # Note: There is room for performance improvements here.  Sort could be replaced
    # with a randomized or deterministic select, if it is known how many results are relevant
    # (knowing how many k slots need to be filled).  These are O(n) operations on average.
    # Or perhaps getting a few minimum/maximum elements O(n).
    # Sorting is likely an O(n log n) operation.
    # Question Classification would have to take over the responsibility of figuring out
    # the k number of slots.
    def string_similarity_score(self, mention) -> List[Tuple[float, str]]:
        entity_candidates = self.index_kg.all_entity_labels()

        Score_Result = namedtuple("Score_STS", 'ld_div_ls, ld, ls, cand')
        # score a single mention and candidate
        def score_sts(m, cand) -> Score_Result:
            ld = edit_distance(" ".join(m), cand)
            ls = len(cand)
            return Score_Result(ld/ls, ld, ls, cand)

        # computed_scores = [(score_sts(mention, cand)) for cand in entity_candidates]
        computed_scores = []
        # print(mention)
        for m in list(mention):
            if self.verbose:
                print(m)
            for cand in entity_candidates:
                if self.verbose:
                    print(f'{m} {cand}')
                computed_scores.append(score_sts(m, cand))

        # 2. Sort STS based on ld/ls
        computed_scores.sort(key=attrgetter('ld_div_ls'))
    
        # 3. Sort STS based on ls descending if ld is equal to zero
#        if any(filter(lambda x: x.ld == 0, computed_scores)):  # this works
        if min(computed_scores, key=attrgetter('ld')) == 0:
            computed_scores.sort(key=attrgetter('ls'), reverse=True)

        if self.verbose:
            print(computed_scores)

        # 4. Compute STR (STring Ranks) including entity candidates' string-ranks where the
        # string-rank of the entity candidate c is 1/index(stc) where index(stc) is the index
        # of the entity candidate c in STS Vector
        def compute_string_ranks(computed_scores: list) -> List[Tuple]:
            st_rank = [(1.0 / i, cs) for i, cs in enumerate(computed_scores, start=1)]
            return st_rank

        st_ranks = compute_string_ranks(computed_scores)
        # 5. Compute vector SES (SEmantic Similarity) containing the cosine similarity cs
        # between the embeddings of an entity mention m and the embeddings of entity candidates
        def compute_semantic_similarity(mention) -> List[Tuple]:
            embedding_mention = [self.sbert_model.encode(' '.join(m), convert_to_tensor=True) for m in mention]
            ses_vec = []
            for emb_m in embedding_mention:
                for label, emb_c in self.entity_candidate_embeddings:
                    cs = util.cos_sim(emb_m, emb_c)
                    ses_vec.append((cs, emb_m, label))

            # this should be the cosine similarity result and the entity candidate label
            return ses_vec

        ses = compute_semantic_similarity(mention)
        # 6. Sort SES based on cs descending
        ses.sort(key=lambda x: x[0], reverse=True)

        # 7. Compute SER (SEmatic Ranks) including the entity candidates' semantic-ranks where
        # the semantic-rank of the entity candidate c is 1/index(sec) where index(sec) is the
        # index of the entity candidate c in SES vector

        # Note is the unpacking ses_scores correctly?
        def compute_semantic_ranks(ses_scores: list) -> List[Tuple]:
            sem_ranks = [(1.0 / i, c) for i, c in enumerate(ses_scores, start=1)]
            return sem_ranks

        sem_ranks = compute_semantic_ranks(ses)
        # 8. Compute similarity score of an entity mention m and the entity candidate c based on
        # the sum of their ranks in STR and SER

        # print out the top match
        # print(f'STR: {st_ranks[0]}')
#        print(f'SES: {sem_ranks[0]}')
        # This shows that there are several scores that match in sem_ranks
        # print([(r[0], r[1][2])for r in sem_ranks])
#        length = [r[1].cand for r in st_ranks]
#        print(len(length))


        similarity_score = defaultdict(lambda: 0)
        # I am going to sum all entries in st_ranks and sem_ranks
#        for score, sts in st_ranks:
#            similarity_score[sts.cand] += score

        for score, ser in sem_ranks:
            # print(f'SER2: {ser[2]}')
            similarity_score[ser[2]] += score

        if self.verbose:
            print(similarity_score)
        similarity_score_list = [(score, label) for label, score in similarity_score.items()]
        similarity_score_list.sort(key=lambda x: x[0], reverse=True)
        if self.verbose:
            print("======== Top Similarity Scores ========")
            for i in range(10):
                print(f"{similarity_score_list[i]}")

        return similarity_score_list

def single_question_test():
    sentence = 'Are auto-dismantling yards permitted?'
    ecl = EntityClassLinking()
    ngram = ecl.ngram_collection(sentence)
    ecl.string_similarity_score(ngram)


# this takes almost 2 hours on CPU for 2711 questions.
def all_question_test():
    from pathlib import Path
    import sys
    import rdflib
    sys.path.append("..")  # allows triplesdb imports
    from triplesdb.generate_template import TemplateGeneration

    kg = rdflib.Graph()
    kg.parse("triplesdb/combined.ttl")
    template_path = Path('../triplesdb/templates')
    tg = TemplateGeneration(template_path)
    tmpls = tg.generate_all_templates(kg, kg)

    ecl = EntityClassLinking()

    # tg = generate_templates()
    # questions = [generated_data['question'] for generated_data in tg]
    questions = [generated_data['question'] for generated_data in tmpls]

    for q in questions:
        ngram = ecl.ngram_collection(q)
        ecl.string_similarity_score(ngram)
        # the answer is not stored


if __name__ == '__main__':
#    ecl = EntityClassLinking(verbose=True)
#    single_question_test()
    all_question_test()
