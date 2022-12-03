"""
Semantic Parsing is a main class for the doing everything need to do training
and classifying for this step.
"""

# Entire classify pipeline takes about 10 seconds CPU time per question.

# Python Standard Libraries
import random
from typing import List, Union

# external library imports
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score

# project internal imports
from question_classification import QuestionClassification
from entity_class_linking import EntityClassLinking
from relation_extraction import RelationExtraction
from slot_filling_query_execution import SlotFillingQueryExecution
# from kg_helper import generate_templates, get_template
import kg_helper


# FIXME: Why is name Class in this class???
class SemanticParsingClass:
    """This is the interface for the Semantic Parsing Step.

    This is the easy way to interact with this class
    """

    def __init__(self):
        self.qc = QuestionClassification()
        self.relex = RelationExtraction()

    def train_all(self):
        """This does all of the training in sequentially."""
#        question_corpus = list(kg_helper.generate_templates())
        # remove questions that are empty sets
#        question_corpus_filt = self._remove_false_answers(self._remove_empty_answers(question_corpus))
        # WORKAROUND: doing this for questions that don't have the correct answers.
        question_corpus_filt = self.generate_filtered_corpus()

        # 1) Question Classification
        # qc = QuestionClassification()
        self.qc.train2(question_corpus_filt)
        # 2) Entity Linking and Class Linking
        # There isn't anything to train for this step.
        # ecl = EntityClassLinking()
        # ngram = ecl.ngram_collection(sentence)
        # similarity_scores = ecl.string_similarity_score(ngram)

        # 3) Relation Extraction
        questions, variables = self.relex.process_question_corpus(question_corpus_filt)
        self.relex.train(questions, variables)

    def classify(self, sentence: str) -> Union[bool, List[str]]:
        # 1) Question Classification
        classified_template_number = self.qc.classify(sentence)
        classified_template_name = self.qc.classification_number_to_template_name(classified_template_number)

        # 2) Entity Linking and Class Linking
        ecl = EntityClassLinking(verbose=False)  # too talkative
        ngram = ecl.ngram_collection(sentence)
        similarity_scores = ecl.string_similarity_score(ngram)

        # 3) Relation Extraction
        # Some questions for Zoning are so simple that they only have one possible predicate,
        # in those case this step can be skipped entirely.  This is the case for most of the questions.
        # About 1/4 of the Zoning questions need this step.
        template_dict = kg_helper.get_template(classified_template_name)
        if template_dict['knowledge_graph'] == 'permitted_uses':
            # skip Relation Extraction
            most_relevant_relations = None
        else:
            # For zoning the k value is 1 (or zero [case above]) meaning that there is only one relationship to find.
            # This is due to the Zoning KG example being very simple, unlike the Tourism KG from the original paper.
            most_relevant_relations = self.relex.extract(sentence, k=1)

        # 4) Slot Filling and Query Execution
        sfqe = SlotFillingQueryExecution()
        return sfqe.slot_fill_query_execute(classified_template_name, similarity_scores, most_relevant_relations)

    def generate_filtered_corpus(self) -> List[dict]:
        question_corpus = list(kg_helper.generate_templates())
        # remove questions that are empty sets
        question_corpus_filt = self._remove_false_answers(self._remove_empty_answers(question_corpus))
        return question_corpus_filt

    # There are some distinctions in :ZoningDistrict and :ZoningDivisionDistrict that did not appear in the original
    # generate_template.  This is causing a few questions that could be answerable, but it would have to use a :seeAlso
    # hop.  This is doable, but out of scope for this phase of the project.  The function below is the workaround.
    def _remove_empty_answers(self, question_corpus: List[dict]) -> List[dict]:
        return [q for q in question_corpus if q['answer'] != []]
        # qc_filtered = []
        # for q in question_corpus:
        #    if q['answer'] == []:
        #        continue
        #    qc_filtered.append(q)
        # return

    # Some questions should be producing True as their output.  They are producing false.  The last minute workaround
    # is to just remove them from the corpus.  generate_template needs to be fixed.
    def _remove_false_answers(self, question_corpus: List[dict]) -> List[dict]:
        return [q for q in question_corpus if q['answer'] != False]

def generate_all_templates_test():
    for res in kg_helper.generate_templates():
        print(res)

# 4 questions take 16 seconds to answer on CPU
def simple_classify_test():

    questions = ['What is the minimum side setback in the R2a zoning district?',  # Works
                 'Are auto-dismantling yards permitted?',                        # works
                 'Which zoning districts allow physical fitness centers?',       # works
                 'Are salt works allowed in a FI3 zoning district?',             # works
                 ]

    sem_par = SemanticParsingClass()
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
def measure_accuracy():
    logger.info("measuring the accuracy of Zoning KGQAS")
    sem_par = SemanticParsingClass()
    question_corpus = list(kg_helper.generate_templates())
    # remove questions that are empty sets and False, this takes it down to 900 questions
    question_corpus_filt = sem_par._remove_false_answers(sem_par._remove_empty_answers(question_corpus))
    print(f'question_corpus_filt: {len(question_corpus_filt)}')
#    SUBSET = 0
    SUBSET = 10
    if SUBSET > 0:
        logger.info(f"measuring subset of size: {SUBSET}")
        answers = [sem_par.classify(q['question']) for q in question_corpus_filt[:SUBSET]]
        gold_answers = [q['answer'] for q in question_corpus_filt[:SUBSET]]
    else:
        answers = [sem_par.classify(q['question']) for q in question_corpus_filt]
        gold_answers = [q['answer'] for q in question_corpus_filt]

    # Need to take lists convert it to a string for accuruacy
#    accuracy = accuracy_score(gold_answers.join, answers)
    # take all the inner lists and join as strings them with commas
    accuracy = accuracy_score([','.join(a) for a in gold_answers], [','.join(a) for a in answers])
    f1 = f1_score(gold_answers, answers, average='micro')
    print(f'# answers: {len(answers)} accuracy:  {accuracy * 100.0} f1 score: {f1 * 100.0}')
    print(answers)

# training time is 2 minutes on CPU
def train_all():
    """This trains both models that have to be trained"""
    sem_par = SemanticParsingClass()
    sem_par.train_all()

if __name__ == '__main__':
#    generate_all_templates_test()
#    simple_classify_test()
    measure_accuracy()
#    train_all()