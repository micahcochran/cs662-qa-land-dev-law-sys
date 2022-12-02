"""
Semantic Parsing is a main class for the doing everything need to do training
and classifying for this step.
"""

# Entire classify pipeline takes about 10 seconds CPU time per question.

# Python Standard Libraries
from typing import List, Union


# project internal imports
from question_classification import QuestionClassification
from entity_class_linking import EntityClassLinking
from relation_extraction import RelationExtraction
from slot_filling_query_execution import SlotFillingQueryExecution
# from kg_helper import generate_templates, get_template
import kg_helper

class SemanticParsingClass:
    """This is the interface for the Semantic Parsing Step.

    This is the easy way to interact with this class
    """

    def __init__(self):
        self.qc = QuestionClassification()
        self.relex = RelationExtraction()


    def train_all(self):
        """This does all of the training in sequentially."""
        question_corpus = list(kg_helper.generate_templates())
        # 1) Question Classification
        # qc = QuestionClassification()
        self.qc.train2(question_corpus)
        # 2) Entity Linking and Class Linking
        # There isn't anything to train for this step.
        # ecl = EntityClassLinking()
        # ngram = ecl.ngram_collection(sentence)
        # similarity_scores = ecl.string_similarity_score(ngram)

        # 3) Relation Extraction
        questions, variables = self.relex.process_question_corpus(question_corpus)
        self.relex.train(questions, variables)


    def classify(self, sentence: str) -> Union[bool, List[str]]:
        # 1) Question Classification
        classified_template_number = self.qc.classify(sentence)
        classified_template_name = self.qc.classification_number_to_template_name(classified_template_number)

        # 2) Entity Linking and Class Linking
        ecl = EntityClassLinking()
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


if __name__ == '__main__':
#    generate_all_templates_test()
    simple_classify_test()