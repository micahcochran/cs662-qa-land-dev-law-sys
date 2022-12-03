"""
Semantic Parsing Phase - 3) Relation Extraction
"""
# internal Python libraries
import itertools
import pickle
import random
from typing import List, Optional, Tuple

# imports from external libraries
from loguru import logger
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer, util

# internal imports
import indexes
# from kg_helper import generate_dim_templates, generate_templates
from kg_helper import generate_dim_templates

# TODO Templates will vary, but any of the permitted use templates, we will already know the predicate for that.
#      This may be able to be skipped entirely for that case.
#

# TODO will have to modify training to split predicates

class RelationExtraction():
    def __init__(self, index_kg: Optional[indexes.IndexesKG] = None):
        # This is a faster model
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        # MLP Classifier model
        self.model = None

        if index_kg is None:
            index_kg = indexes.IndexesKG()

        self.index_kg = index_kg
        # below code create a 1-hot encoding for labels
#        self.mlb = MultiLabelBinarizer(classes=sorted(idx.all_index_labels()))
        self.mlb = MultiLabelBinarizer(classes=sorted(index_kg.predicate_labels()))

#        _, variables = self.process_question_corpus(list(generate_templates()))
#        self.label_encoding = self.mlb.fit_transform(variables)
        # Why is the pre_generating label encoding????
        _, variables = self.process_question_corpus(list(generate_dim_templates()), verbose=False)
#        print(f"variables: {variables}")
#        # filter out variables that are not in the predicates
        variables_filtered = [self.filter_relation_variables(row) for row in variables]
#        print(f"variables_filtered: {variables_filtered}")
        self.label_encoding = self.mlb.fit_transform(variables_filtered)
#        print(f"self.label_encoding: {self.label_encoding}")
#        self.label_encoding = self.mlb.fit_transform(index_kg.predicate_labels())

    def filter_relation_variables(self, variables: List[str]) -> List[str]:
        """filter out variables that are not in the predicates"""
        return [v for v in variables if v in self.index_kg.predicate_labels()]

    def process_question_corpus(self, question_corpus, verbose=True) -> Tuple[List[str], List[List[str]]]:
        """Takes the question corpus and splits it into parallel lists of questions and variable values used to fill in
        the question.
        return (questions, variables)"""
        def split_question_and_variables(question_corpus):
           return [(question['question'], list(question['variables'].values())) for question in question_corpus]
        # this is just for debug
        questions_and_variables = split_question_and_variables(question_corpus)

        questions = [question['question'] for question in question_corpus]
        variables_intermediate = [list(question['variables'].values()) for question in question_corpus]


        # remove strings that begin with : or [ ...  URI fragment predicates ":maxBulidingHeight" and units "[ft_i]"
        # These are useless to the production of the text version of the question
        # (perhaps the generate_template.py needs to be rewritten to not
        # have these values for the question text.)  This is not idea           .
        variables = [list(filter(lambda x: x[0] not in (':', '['), vi))
                                    for vi in variables_intermediate]

        # variables = list(filter(lambda v: v in self.index_kg.predicate_labels(), variables_no_units_frags))
        if verbose:
            print(f"{len(questions_and_variables)}, {len(questions)}, {len(variables)}")
            print("===== 10 Random Question and Variables ===== ")
            for i in range(10):
                rnd = random.randint(0, len(questions_and_variables))
                # print(questions_and_variables[rnd])
                print(f"Question: {questions[rnd]}")
                print(f"Variables: {variables[rnd]}")
        return questions, variables

    def train(self, questions: List[str], variables: List[List[str]]):

        # label_encoding = self.mlb.fit_transform(variables)
        # mlb.fit([idx.all_index_labels()])
#        q_embeddings = [self.sbert_model.encode(q, convert_to_tensor=True) for q in questions]
        # Use SBERT to generate the embedding for the MLP
        q_embeddings = [self.sbert_model.encode(q) for q in questions]

        # print(q_embeddings)
#        print(self.label_encoding)
        variables_encoded = self.mlb.fit_transform(variables)

#        X_train, X_test, y_train, y_test = train_test_split(q_embeddings, self.label_encoding, random_state=246341428,
#                                                            shuffle=True)
        X_train, X_test, y_train, y_test = train_test_split(q_embeddings, variables_encoded, random_state=246341428,
                                                            shuffle=True)

        # Used Table 5. MLP Parameters from paper
        params = {
            'activation': 'logistic',  # uses sigmoid activation function
            'hidden_layer_sizes': (768, 29),  # ????????????
            'max_inter': 100,       # epochs
            'learning_rate_init': 0.01,
            'solver': 'adam',
        }
        # NOTE: Not sure how to specify these things:
        # Input Dimensions: 768
        # Loss Functions: binary_crossentropy
        # Output Dimensions: 29
        # 1 input layer, 2 hidden layers, one output layer
        # VERBOSITY
        # Printout of Accuracy

#        print("Embedding first 10: ")
#        print(q_embeddings)
#        print("Labels: ")
#        print(variables)

        # the parameters are a little different from the scikit learn version.
 #       model = MLPClassifier(*params)
        #
        # max_iter=100,
        # hidden_layer_sizes=(768, 114),
        # hidden_layer_sizes=(768, 100, 100, 29),

        # 8.35 no answers correct
#        self.model = MLPClassifier(activation='logistic', learning_rate='adaptive', max_iter=100, hidden_layer_sizes=(768, 100, 100, 29),
#                                   learning_rate_init=0.01, solver='adam', random_state=246341428, verbose=True)
        # 8.35
#        self.model = MLPClassifier(activation='logistic', learning_rate='adaptive', max_iter=100, hidden_layer_sizes=(768, 384, 192, 29),
#                                   learning_rate_init=0.01, solver='adam', random_state=246341428, verbose=True)

#        self.model = MLPClassifier(activation='logistic', max_iter=100, hidden_layer_sizes=(768, 384, 192, 29),
#                                   learning_rate_init=0.01, solver='adam', random_state=246341428, verbose=True)
        # gets closer to zero, but the loss gets to 0.475
#        self.model = MLPClassifier(max_iter=500, random_state=246341428, verbose=True)

        # does not reach convergence, loss 4.9
#        self.model = MLPClassifier(activation='logistic', max_iter=500, random_state=246341428, verbose=True)
        # does not convert, lost 0.5 -- works fairly well
        self.model = MLPClassifier(solver='adam', max_iter=500, random_state=246341428, verbose=True)

        # using the defaults - no correct answers
#        self.model = MLPClassifier(random_state=246341428, verbose=True)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
#        predictions = [round(value) for value in y_pred]
#        accuracy = accuracy_score(y_test, predictions)
#        f1 = f1_score(y_test, y_pred, average='micro')
        # SCORING IS HAVING ISSUES
        # for scoring it throws ValueError: X has 9 features, but MLPClassifier is expecting 384 features as input.
#        accuracy = self.model.score(y_test, y_pred)
#        print(f"Accuracy {accuracy * 100.0}")
#        print(f"F1 Score: {f1* 100.0}")
        with open("relation_extraction_model.pickle", "wb") as fp:
            pickle.dump(self.model, fp)
        return self.model


    # What are the parameters need to fill in extract
    def get_parameters(self):
        pass
        # Needs to do the following:
        # match the number from question_classification to a template
        # figure out the number of k spots to fill
        # MLP Classifier will tell you if it is correct or not

    def extract(self, sentence: str, k: int) -> np.ndarray:
        """
        :param sentence:
        :param k: the number of spots to be extracted
        :return:
        """
        # Note: k should not be 0, that means there is no work to do.

        if self.model is None:
            logger.info('Loading MLPClassifier Model: relation_extraction_model.pickle')
            with open("relation_extraction_model.pickle", 'rb') as pick_in:
                self.model = pickle.load(pick_in)

        # find top-k relations selected , k is the number of required relation holders in the assigned query pattern

        embedding = self.sbert_model.encode(sentence)
        reshaped = embedding.reshape(1, -1)
        # Predict the probability for each class label
        result = self.model.predict_proba(reshaped)
        # result_labels = self.mlb.inverse_transform(result)
        # print(result)
        # extract the top-k relations
        p_result = pd.Series(result[0])
        k_results = p_result.nlargest(k)
        return self.mlb.classes_[k_results.index]  # array of most likely matches

        # return result_labels  # this is a list of string
        # return result  # this is a matrix

        # Use SBERT to generate the embedding for the MLP
#        embedding = self.sbert_model.encode(sentence, convert_to_tensor=True)

        # Used Table 5. MLP Parameters from paper
 #       params = {
 #           'activation': 'logistic',  # uses sigmoid activation function
 #           'hidden_layer_sizes': (100, 100),  # ????????????
 #           'max_inter': 100,       # epochs
 #           'learning_rate': 0.01,
 #           'solver': 'adam',
 #       }
        # NOTE: Not sure how to specify these things:
        # Input Dimensions: 768
        # Loss Functions: binary_crossentropy
        # Output Dimensions: 29
        # 1 input layer, 2 hidden layers, one output layer


        # the parameters are a little different from the scikit learn version.
#        clf = MLPClassifier(*params)
#        clf.fit(embedding, )

        # find top-k relations selected , k is the number of required relation holders in the assigned query pattern
#        if self.model is None:
#            logger.info('Loading MLPClassifier Model: relation_extraction_model.pickle')
 #           self.model = MLPClassifier()
#            with open("relation_extraction_model.pickle") as fp:
#                pickle.load(self.model, fp)


# Here are a few examples questions that are passed through:

# Is the minimum lot size for a property in the C1 zoning district 6000 square feet?
# ['minimum lot size', '6000', 'C1', 'square feet']
# ('Are libraries allowed in a C1 zoning district?', ['libraries', 'C1'])

# Question: What is the minimum lot width in the R3a zoning district?
# Variables: ['minimum lot width', 'R3a']
# Question: I would like to build public parking lots.  Which zoning districts permits this use?
# Variables: ['public parking lots']

### Nonsense question from generate_template.py:
# Question: What is the permits use in the C4 zoning district?
# Variables: ['permits use', 'C4']


def main_training():
    relex = RelationExtraction()
#    questions, variables = relex.process_question_corpus(list(generate_templates()))
    questions, variables = relex.process_question_corpus(list(generate_dim_templates()))
    relex.train(questions, variables)

def small_training_test():
    """small test of main for debugging"""
    ex_questions = [
        'I would like to build public parking lots.  Which zoning districts permits this use?',
        'Are libraries allowed in a C1 zoning district?',
        'Is the minimum lot size for a property in the C1 zoning district 6000 square feet?',
        'What is the minimum lot width in the R3a zoning district?',
    ]
    ex_variables = [
        ['public parking lots'],
        ['libraries', 'C1'],
        ['minimum lot size', '6000', 'C1', 'square feet'],
        ['minimum lot width', 'R3a']
    ]

    relex = RelationExtraction()
    relex.train(ex_questions, ex_variables)

def extract_test(relex=None):
    ex_questions = [
#        'I would like to build public parking lots.  Which zoning districts permits this use?',
#        'Are libraries allowed in a C1 zoning district?',
        # the below question is having problems picking up '6000'
        'Is the minimum lot size for a property in the C1 zoning district 6000 square feet?',
        'What is the minimum lot width in the R3a zoning district?',
    ]
    ex_variables = [
#        ['public parking lots'],   # no relations
#        ['libraries', 'C1'],       # no relations
        ['minimum lot size', '6000', 'C1', 'square feet'],
        ['minimum lot width', 'R3a']
    ]
    if relex is None:
        relex = RelationExtraction()

    for i, q in enumerate(ex_questions):
        k = len(relex.filter_relation_variables(ex_variables[i]))
        print(f"Question {i}: {q}, k: {k}")
        result = relex.extract(q, k)
        print(result)


# This process takes 15 seconds for 270 questions
# SCORING NOT WORKING
def extract_all_test(relex=None):
    if relex is None:
        relex = RelationExtraction()

    questions, gold_variables = relex.process_question_corpus(question_corpus=list(generate_dim_templates()))
    # gold_encoding = relex.label_encoding   # accessing class variables is a BAD idea
    for i, q in enumerate(questions):
        k = len(relex.filter_relation_variables(gold_variables[i]))
        relex.extract(q, k)

    # f1 = f1_score(gold_encoding, results)
    # print(f"F1 Score: {f1 * 100.0}")


if __name__ == '__main__':
#    extract_all_test()
#    relex = main_training()
#    extract_test(relex)

    extract_all_test()
#    extract_test()
#    small_training_test()