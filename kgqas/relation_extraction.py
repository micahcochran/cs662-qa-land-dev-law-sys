"""
Semantic Parsing Phase - 3) Relation Extraction
"""
# internal Python libraries
import pickle
import random
from typing import List, Tuple

# imports from external libraries
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer, util

# internal imports
from indexes import IndexesKG

class RelationExtraction():
    def __init__(self):
        # This is a faster model
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        # MLP Classifier model
        self.model = None

    def process_question_corpus(self, question_corpus) -> Tuple[List[str], List[List[str]]]:
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
        # have these values for the question text.)  This is not idea.
        variables = [list(filter(lambda x: x[0] not in (':', '['), vi))
                        for vi in variables_intermediate]

        print(f"{len(questions_and_variables)}, {len(questions)}, {len(variables)}")
        print("===== 10 Random Question and Variables ===== ")
        for i in range(10):
            rnd = random.randint(0, len(questions_and_variables))
            # print(questions_and_variables[rnd])
            print(f"Question: {questions[rnd]}")
            print(f"Variables: {variables[rnd]}")
        return questions, variables

    def train(self, questions: List[str], variables: List[List[str]]):
        # Use SBERT to generate the embedding for the MLP
        # some of the variable names need to be removed.

        mlb = MultiLabelBinarizer()
        idx = IndexesKG()
        mlb.fit([idx.all_index_labels()])
        mlb.fit_transform(variables)
#        q_embeddings = [self.sbert_model.encode(q, convert_to_tensor=True) for q in questions]
        q_embeddings = [self.sbert_model.encode(q) for q in questions]

        X_train, X_test, y_train, y_test = train_test_split(q_embeddings, variables, random_state=246341428,
                                                            shuffle=True)
        # Used Table 5. MLP Parameters from paper
        params = {
            'activation': 'logistic',  # uses sigmoid activation function
 #           'hidden_layer_sizes': (100, 100),  # ????????????
            'max_inter': 100,       # epochs
            'learning_rate_init': 0.01,
            'solver': 'adam',
        }
        # NOTE: Not sure how to specify these things:
        # Input Dimensions: 768
        # Loss Functions: binary_crossentropy
        # Output Dimensions: 29
        # 1 input layer, 2 hidden layers, one output layer

#        print("Embedding first 10: ")
#        print(q_embeddings)
#        print("Labels: ")
#        print(variables)

        # the parameters are a little different from the scikit learn version.
 #       model = MLPClassifier(*params)
        self.model = MLPClassifier(activation='logistic', max_iter=100, learning_rate_init=0.01, solver='adam')

        self.model.fit(X_train, mlb)

        y_pred = self.model.predict(X_test)
        predictions = [round(value) for value in y_pred]
#        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='micro')
#        print(f"Accuracy {accuracy * 100.0}")
        print(f"F1 Score: {f1* 100.0}")
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
    def extract(self, sentence: str):
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
        if self.model is None:
            logger.info('Loading MLPClassifier Model: relation_extraction_model.pickle')
 #           self.model = MLPClassifier()
            with open("relation_extraction_model.pickle") as fp:
                pickle.load(self.model, fp)


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


def main_test():
    relex = RelationExtraction()
    questions, variables = relex.process_question_corpus(list(generate_templates()))
    relex.train(questions, variables)

def small_test():
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


if __name__ == '__main__':
#    main_test()
    small_test()