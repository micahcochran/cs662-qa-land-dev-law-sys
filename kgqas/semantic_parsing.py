"""
Semantic Parsing is a main class for the doing everything need to do training
and classifying for this step.
"""

# Entire classify pipeline takes about 10 seconds CPU time per question.

# Python Standard Libraries
from pathlib import Path
from pprint import pprint
import random
import sys
import time
from typing import List, Optional, Union

# external library imports
from loguru import logger
import rdflib
from sklearn.metrics import accuracy_score, f1_score

# project internal imports
from question_classification import QuestionClassification
from entity_class_linking import EntityClassLinking
from relation_extraction import RelationExtraction
from slot_filling_query_execution import SlotFillingQueryExecution

# internal libraries that need a different path
sys.path.append("..")
from triplesdb.generate_template import TemplateGeneration



# FIXME: Why is name Class in this class???
class SemanticParsingClass:
    """This is the interface for the Semantic Parsing Step.

    This is the easy way to interact with this class
    """

    def __init__(self, template_generation: Optional[TemplateGeneration] = None,
                    knowledge_graph: Optional[rdflib.Graph] = None, verbose = False):

        self.verbose = verbose

        if knowledge_graph is None:
            self.kg = rdflib.Graph()
            self.kg.parse("triplesdb/combined.ttl")
        else:
            self.kg = knowledge_graph

        if template_generation is None:
            template_path = Path('../triplesdb/templates')
            self.tg = TemplateGeneration(template_path)
        else:
            self.tg = template_generation

        self.qc = QuestionClassification(template_generation=self.tg, knowledge_graph=self.kg)
        self.relex = RelationExtraction(template_generation=self.tg, knowledge_graph=self.kg)


    def train_all(self):
        """This does all of the training sequentially."""
        question_corpus = list(self.tg.generate_all_templates_shuffle(self.kg, self.kg))
        # print(f'question_corpus: {question_corpus}')

        # 1) Question Classification
        self.qc.train2(question_corpus)

        # 2) Entity Linking and Class Linking
        # There isn't anything to train for this step.

        # 3) Relation Extraction
        questions, variables = self.relex.process_question_corpus(question_corpus)
        self.relex.train(questions, variables)


    def classify(self, sentence: str) -> (Union[bool, List[str]], dict):

        if self.verbose is True:
            logger.debug(f"QUESTION: {sentence}")
        msg = {}

        # 1) Question Classification
        qc_time_start = time.time()
        classified_template_number = self.qc.classify(sentence)
        qc_time_end = time.time()
        classified_template_name = self.qc.classification_number_to_template_name(classified_template_number)
        msg['classified_template_number'] = classified_template_number
        msg['classified_template_name'] = classified_template_name
        msg['qc_time'] = qc_time_end - qc_time_start

        # 2) Entity Linking and Class Linking
        ecl_time_start = time.time()
        ecl = EntityClassLinking(verbose=False)  # too talkative
        ngram = ecl.ngram_collection(sentence)
        similarity_scores = ecl.string_similarity_score(ngram)
        ecl_time_end = time.time()
        msg['similiarity_scores_top_10'] = similarity_scores[:10]
        msg['ecl_time'] = ecl_time_end - ecl_time_start

        # 3) Relation Extraction
        # Some questions for Zoning are so simple that they only have one possible predicate,
        # in those case this step can be skipped entirely.  This is the case for most of the questions.
        # About 1/4 of the Zoning questions need this step.
        template_dict = self.tg.get_template(classified_template_name)
        if template_dict['knowledge_graph'] == 'permitted_uses':
            # skip Relation Extraction
            most_relevant_relations = None
        else:
            # For zoning the k value is 1 (or zero [case above]) meaning that there is only one relationship to find.
            # This is due to the Zoning KG example being very simple, unlike the Tourism KG from the original paper.
            relex_time_start = time.time()
            most_relevant_relations = self.relex.extract(sentence, k=1)
            relex_time_end = time.time()
            msg['relex_time'] = relex_time_end - relex_time_start

        if most_relevant_relations is not None:
            msg['most_relevant_relations'] = most_relevant_relations

        # 4) Slot Filling and Query Execution
        sfqe = SlotFillingQueryExecution(template_generation=self.tg)
        # responsibilities of each function has been reduced
        sf_msg = sfqe.slot_fill(classified_template_name, similarity_scores, most_relevant_relations)
        msg.update(sf_msg)
#            pprint(msg)
        sparql_msg = sfqe.fill_sparql_template(classified_template_name, msg)
        msg.update(sparql_msg)
        answer, qe_msg = sfqe.query_execution(classified_template_name, msg)
        msg.update(qe_msg)
#            answer, slot_msg = sfqe.slot_fill_query_execute(classified_template_name, similarity_scores, most_relevant_relations)
#            msg.update(slot_msg)

        return answer, msg


    # There are some distinctions in :ZoningDistrict and :ZoningDivisionDistrict that did not appear in the original
    # generate_template.  This is causing a few questions that could be answerable, but it would have to use a :seeAlso
    # hop.  This is doable, but out of scope for this phase of the project.  The function below is the workaround.
#    def _remove_empty_answers(self, question_corpus: List[dict]) -> List[dict]:
#        return [q for q in question_corpus if q['answer'] != []]



def generate_all_templates_test():
    kg = rdflib.Graph()
    kg.parse("triplesdb/combined.ttl")
    template_path = Path('../triplesdb/templates')
    tg = TemplateGeneration(template_path)

    for res in tg.generate_all_templates(kg, kg):
        print(res)

def generate_all_templates_shuffle_test():
    kg = rdflib.Graph()
    kg.parse("triplesdb/combined.ttl")
    template_path = Path('../triplesdb/templates')
    tg = TemplateGeneration(template_path)

#    for res in tg.generate_all_templates(kg, kg):
    for res in tg.generate_all_templates_shuffle(kg, kg):
        print(res)


# 4 questions take 16 seconds to answer on CPU
def simple_classify_test():

    questions = ['What is the minimum side setback in the R2a zoning district?',  # Works
                 'Are auto-dismantling yards permitted?',                        # works
                 'Which zoning districts allow physical fitness centers?',       # misclassifies as a yes/no question
                 # above question causes a KeyError for regulation_predicate in slot_fill_query_execute()
                 'Are salt works allowed in a FI3 zoning district?',             # works
                 ]

    sem_par = SemanticParsingClass()
    # should catch KeyError exception in list comprehension from .classify()
    answers = [sem_par.classify(q) for q in questions]

    for q, a in zip(questions, answers):
        print(f"Question: {q}")
        print(f"Answer: {a}")


def get_random_questions_answers(question_corpus: List[dict], n: int) -> List[dict]:
    """Gets n number of random questions and answers from the corpus.
    This is another way to subset, but these are random so you could get the same answers."""
    results = []
    for i in range(n):
        rnd = random.randint(0, len(question_corpus))
        candidate = question_corpus[rnd]
        row = {'question': candidate['question'], 'answer': candidate['answer']}
        results.append(row)
    return results


def _dict_keysorted_string(d: dict) -> str:
    """
    Convert a dictionary to a string like str(), but sorts by key name.
    """
    out = '{'
    for k in sorted(d.keys()):
        if out[-1] != '{':
            out += ", "
        out += f"{k}: {d[k]}"

    out += '}'
    return out

# measures the accuracy of slot filling and answering
# This should take about 2-3 hours for 2700 questions
def measure_accuracy(subset: int = 0, random_state: Optional[int] = None):
    """measure the accuracy and F1-score of the Slot filling and answering
    subset - measure a subset of the questions, numeric value of the number of questions to test
    random_state - use numeric value to randomize questions"""
    logger.info("measuring the accuracy of Zoning KGQAS")
    start_time = time.time()
    sem_par = SemanticParsingClass()

    kg = rdflib.Graph()
    kg.parse("triplesdb/combined.ttl")
    template_path = Path('../triplesdb/templates')
    tg = TemplateGeneration(template_path)
    # FIXME: generate_all_templates_shuffle() random_state is not quite repeatable
    question_corpus = list(tg.generate_all_templates_shuffle(kg, kg, random_state=random_state))

    if subset > 0:
        logger.info(f"measuring subset of size: {subset}")
        answers_message = [sem_par.classify(q['question']) for q in question_corpus[:subset]]
        msgs = [msg for a, msg in answers_message]
        answer_slots_filled = [msg['filled_slots'] for msg in msgs]
        gold_answers = [q['answer'] for q in question_corpus[:subset]]
        gold_slots = [q['variables'] for q in question_corpus[:subset]]
    else:
        logger.info(f'measuring entire corpus of size: {len(question_corpus)}')
        answers_message = [sem_par.classify(q['question']) for q in question_corpus]
        msgs = [msg for a, msg in answers_message]
        answer_slots_filled = [msg['filled_slots'] for msg in msgs]
        gold_answers = [q['answer'] for q in question_corpus]
        gold_slots = [q['variables'] for q in question_corpus]

    answers = [a for a, msg in answers_message]

#    print('$$$$$ ANSWER SLOTS FILLED $$$$$')
#    pprint(answer_slots_filled)
#    print('$$$$$$$$ GOLD SLOTS $$$$$$$')
#    pprint(gold_slots)
#    print('$$$$$$$$$$$$$$$  MESSAGES  $$$$$$$$$$$$$$$$')
#    pprint(msgs)

    # mass filter of keys, I would prefer something a little better like getting the keys from the template
    # regulation_predicate should not be in 
    # Another ways to do this in Python 3.11 is to use string.Template.get_identifiers() from the template.
    FILTERED_KEYS = ('regulation_number', 'regulation_predicate', 'use', 'unit_datatype',
                     'unit_symbol', 'zoning_dims', 'zoning')


    def flatten_lists(x: List[str]) -> str:
        """flattens a list from ['a', 'b', 'c'] to 'a, b, c' """
        if isinstance(x, list):
            return ', '.join(x)
        return x

    # slot answer conversion
    def answer_conversion(a, filter_keys: bool=False):
        if isinstance(a, dict):
            if filter_keys is True:
                filtered = {k: v for k, v in a.items() if k in FILTERED_KEYS} 
                return _dict_keysorted_string(filtered)
            else:
                return _dict_keysorted_string(a)
        else:
            return a

    # take all the inner lists and join their strings them with commas - REMOVE
    gold_answer_slots_sortedstrs = [answer_conversion(a, True) for a in answer_slots_filled]
    # Note: the order should be the same due to using the same code to get there. 
    #       There doesn't seem to be a need to sort the results.  - REMOVE
    answer_slots_sortedstrs = [answer_conversion(a) for a in answer_slots_filled]

    accuracy_sf = accuracy_score(gold_answer_slots_sortedstrs, 
                                 answer_slots_sortedstrs)


    # take all the inner lists and join their strings them with commas
    gold_answers_flattened = [flatten_lists(a) for a in gold_answers]
    # Note: the order should be the same due to using the same code to get there. 
    #       There doesn't seem to be a need to sort the results.
    answers_flattened = [flatten_lists(a) for a in answers]

    # print(f'gold_answers_flattened: {gold_answers_flattened}')
    # print(f'answers_flattened: {answers_flattened}')

    accuracy = accuracy_score(gold_answers_flattened, 
                              answers_flattened)

    f1_sf = f1_score(gold_answer_slots_sortedstrs, answer_slots_sortedstrs, average='micro')

    f1 = f1_score(gold_answers_flattened, answers_flattened, average='micro')
     
    print(f'# answers: {len(answers)} Answer accuracy:  {accuracy:.2%}, Answer f1 score: {f1:.2%}')
    print(f'Slot Filled accuracy:  {accuracy_sf:.2%}, Slot Filled f1 score: {f1_sf:.2%}')
    runtime = time.time()-start_time
    print(f'Runtime: {runtime:.5} s, per question runtime {runtime/len(answers)}')

#    print(answers)




# training time is 9 minutes on CPU
def train_all():
    """This trains both models that have to be trained"""
    sem_par = SemanticParsingClass()
    sem_par.train_all()


if __name__ == '__main__':
#    generate_all_templates_test()

#    simple_classify_test()
    # 90 questions take about 6 minutes CPU
#    measure_accuracy(90)
#    measure_accuracy(10)


#    train_all()

    measure_accuracy(5, random_state=42)