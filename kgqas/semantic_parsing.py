"""
Semantic Parsing is a main class for the doing everything need to do training
and classifying for this step.
"""

# Entire classify pipeline takes about 10 seconds CPU time per question.

# Python Standard Libraries
from pathlib import Path
import random
import sys
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


# from kg_helper import generate_templates, get_template
# import kg_helper
# internal libraries that need a different path
sys.path.append("..")
from triplesdb.generate_template import TemplateGeneration



# FIXME: Why is name Class in this class???
class SemanticParsingClass:
    """This is the interface for the Semantic Parsing Step.

    This is the easy way to interact with this class
    """

    def __init__(self, template_generation: Optional[TemplateGeneration] = None,
                    knowledge_graph: Optional[rdflib.Graph] = None):

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
        logger.debug(f"QUESTION: {sentence}")
        msg = {}

        # 1) Question Classification
        classified_template_number = self.qc.classify(sentence)
        classified_template_name = self.qc.classification_number_to_template_name(classified_template_number)
        msg['classified_template_number'] = classified_template_number
        msg['classified_template_name'] = classified_template_name

        # 2) Entity Linking and Class Linking
        ecl = EntityClassLinking(verbose=False)  # too talkative
        ngram = ecl.ngram_collection(sentence)
        similarity_scores = ecl.string_similarity_score(ngram)
        msg['similiarity_scores_top_10'] = similarity_scores[:10]

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
            most_relevant_relations = self.relex.extract(sentence, k=1)

        if most_relevant_relations is not None:
            msg['most_relevant_relations'] = most_relevant_relations

        # 4) Slot Filling and Query Execution
        sfqe = SlotFillingQueryExecution(template_generation=self.tg)
        answer, slot_msg = sfqe.slot_fill_query_execute(classified_template_name, similarity_scores, most_relevant_relations)
        msg.update(slot_msg)
        return answer, msg

#    def generate_filtered_corpus(self) -> List[dict]:
#        question_corpus = list(kg_helper.generate_templates())
        # remove questions that are empty sets
#        question_corpus_filt = self._remove_false_answers(self._remove_empty_answers(question_corpus))
#        return question_corpus_filt


    # There are some distinctions in :ZoningDistrict and :ZoningDivisionDistrict that did not appear in the original
    # generate_template.  This is causing a few questions that could be answerable, but it would have to use a :seeAlso
    # hop.  This is doable, but out of scope for this phase of the project.  The function below is the workaround.
#    def _remove_empty_answers(self, question_corpus: List[dict]) -> List[dict]:
#        return [q for q in question_corpus if q['answer'] != []]

    # Some questions should be producing True as their output.  They are producing false.  The last minute workaround
    # is to just remove them from the corpus.  generate_template needs to be fixed.
#    def _remove_false_answers(self, question_corpus: List[dict]) -> List[dict]:
#        return [q for q in question_corpus if q['answer'] != False]


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


# This takes 44 minutes to run on all questions

# I estimate that this will take about 8 hours for the ~5000 questions.
def measure_accuracy(subset: int = 0, randomized_subset: bool = False):
    """measure the accuracy and F1-score of the Semantic Parsing Process"""
    logger.info("measuring the accuracy of Zoning KGQAS")
    sem_par = SemanticParsingClass()

    kg = rdflib.Graph()
    kg.parse("triplesdb/combined.ttl")
    template_path = Path('../triplesdb/templates')
    tg = TemplateGeneration(template_path)
    question_corpus = list(tg.generate_all_templates_shuffle(kg, kg))
    question_corpus_filt = question_corpus  # FIXME filtering may still be needed but less extensively

    # remove questions that are empty sets and False, this takes it down to 900 questions
#    question_corpus_filt = list(sem_par._remove_false_answers(sem_par._remove_empty_answers(question_corpus)))
#    question_corpus_filt = list(kg_helper.remove_empty_answers(question_corpus))
    print(f'len(question_corpus_filt): {len(question_corpus_filt)}')

    if subset > 0:
        if randomized_subset:
            # randomize in order to try to see if it has problems on certain questions
            logger.info(f"measuring randomized subset of size: {subset}")
            answers = []
            gold_answers = []
            for i in range(subset):
                rnd = random.randint(0, subset)
                a, msg = sem_par.classify(question_corpus_filt[rnd]['question'])
                answers.append(a)
                ga = question_corpus_filt[rnd]['answer']
                gold_answers.append(ga)
        else:
            logger.info(f"measuring subset of size: {subset}")
            answers = [sem_par.classify(q['question']) for q in question_corpus_filt[:subset]]
            gold_answers = [q['answer'] for q in question_corpus_filt[:subset]]
    else:
        answers = [sem_par.classify(q['question']) for q in question_corpus_filt]
        gold_answers = [q['answer'] for q in question_corpus_filt]

    # Need to take lists convert it to a string for accuracy
#    accuracy = accuracy_score(gold_answers.join, answers)

   #  for a in answers:
   #     print(f'a: {a}')
   #     print(f'type of a: {type(a)}')

    def flatten_lists(x: List[str]) -> str:
        """flattens a list from ['a', 'b', 'c'] to 'a, b, c' """
        if isinstance(x, list):
            return ', '.join(x)
        return x
    # that didn't work because it would try to join boolean answers.
    # accuracy = accuracy_score([','.join(a) for a in gold_answers], [','.join(a) for a in answers])

    # take all the inner lists and join their strings them with commas
    gold_answers_flattened = [flatten_lists(a) for a in gold_answers]
    # Note: the order should be the same due to using the same code to get there. 
    #       There doesn't seem to be a need to sort the results.
    answers_flattened = [flatten_lists(a) for a in answers]
    accuracy = accuracy_score(gold_answers_flattened, 
                              answers_flattened)

    f1 = f1_score(gold_answers_flattened, answers_flattened, average='micro')
    print(f'# answers: {len(answers)} accuracy:  {accuracy * 100.0}, f1 score: {f1 * 100.0}')

    print(answers)


# training time is 9 minutes on CPU
def train_all():
    """This trains both models that have to be trained"""
    sem_par = SemanticParsingClass()
    sem_par.train_all()

if __name__ == '__main__':
#    generate_all_templates_test()

#    simple_classify_test()
    measure_accuracy(90)
#    measure_accuracy(5)


#    train_all()