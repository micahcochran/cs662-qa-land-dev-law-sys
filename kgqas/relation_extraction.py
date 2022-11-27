"""
Semantic Parsing Phase - 3) Relation Extraction
"""
# internal Python libraries
import pickle

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
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        # MLP Classifier model
        self.model = None


    def train(self, question_corpus):
        # Use SBERT to generate the embedding for the MLP
        def split_question_and_variables(question_corpus):
            return [(question['question'], list(question['variables'].values())) for question in question_corpus]
        questions_and_variables = split_question_and_variables(question_corpus)
        questions = [qv[0] for qv in questions_and_variables]
        variables = [qv[1] for qv in questions_and_variables]
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

        print("Embedding: ")
        print(q_embeddings)
        print("Labels: ")
        print(variables)

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
#        self.model.save_model("relation_extraction_model.ubj")
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

def main_test():
    # from question_classification import QuestionClassification
    # from entity_class_linking import EntityClassLinking
    from semantic_parsing import generate_templates

    # qc = QuestionClassification()
    relex = RelationExtraction()
    relex.train(generate_templates())


if __name__ == '__main__':
    main_test()