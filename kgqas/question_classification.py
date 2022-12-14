"""
Semantic Parsing Phase - 1) Question Classification
"""

# training takes ~3 minutes on CPU

# Python library imports
from pathlib import Path
import sys
import time
from typing import List, Optional

# External library imports
# install loguru
from loguru import logger
from nltk.tag.perceptron import PerceptronTagger
from nltk.tokenize import word_tokenize
import numpy as np
import rdflib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import spacy
import stanza
# install xgboost
import xgboost as xgb

# internal library imports
from kg_helper import generate_templates, template_names, template_number_dictionary

# internal libraries that need a different path
sys.path.append("..")  # hack to allow triplesdb imports
from triplesdb.generate_template import TemplateGeneration

# The below aren't needed for nltk because I am using stanza to do POS and dependency parsing.
# Run these one time only
# import nltk
# nltk.download('universal_tagset')    # brown when tagset='universal'
# nltk.download('averaged_perceptron_tagger')    # PerceptronTagger(load=True)


# TODO: move the model as a part of the class so it isn't being loaded every time.
class QuestionClassification:
    def __init__(self, template_generation: Optional[TemplateGeneration] = None,
                 knowledge_graph: Optional[rdflib.Graph] = None):
        self.nlp = self.dependency_parse_stanza_initialize()
        self.model = self.load_model()


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

    # UNUSED
    def pos_tagging(self, sentence: str):
        avg_tagger = PerceptronTagger(load=True)
        tokens = word_tokenize(sentence)
        avg_perceptron_tags = avg_tagger.tag(tokens)
        return avg_perceptron_tags

    # NOT WORKING: Had problems finding the POS tag of the headword.
    # result in a triplets in the following form:
    # <POS tag of headword, dependency tag, POS tag of dependent word>
    def dependency_parsing_spacy(self, sentence: str) -> List[tuple]:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentence)
        dependency_triplets = []
        # see, https://spacy.io/api/token for explanation
        for token in doc:
#            print(type(token))
#            print(
#                f'{token.text:{12}} {token.pos_:{6}} {token.tag_:{6}} {token.dep_:{6}} {spacy.explain(token.pos_):{20}} {spacy.explain(token.tag_)}')
            print(
               f'{token.text:{12}} {token.pos_:{6}} <{token.tag_:{6}} {token.dep_:{6}}> {token.head} {spacy.explain(token.pos_):{20}} {spacy.explain(token.tag_)}')
            dependency_triplets.append((doc[token.i].tag_, token.head, token.dep_, token.tag_))
        return [(doc[token.i].tag_, token.head, token.dep_, token.tag_) for token in doc]

    # result in a triplets in the following form:
    # <POS tag of headword, dependency tag, POS tag of dependent word>
    # This WORKS
    # This uses a Standford parser.
    # Documentation: for Stanza Dependency parsing: https://stanfordnlp.github.io/stanza/depparse.html
    def dependency_parse_stanza(self, nlp, sentence: str) -> List[tuple]:
        doc = nlp(sentence)

        dependency_triplets = []
        for sent in doc.sentences:
            for word in sent.words:
                # ignore first word for creating dependency triples
                if word.head > 0:
                    # print(f'{sent.words[word.head-1].xpos}\tword.head:{word.head}\tdeprel: {word.deprel}\txpos: {word.xpos}')
                    dependency_triplets.append((sent.words[word.head-1].xpos, word.deprel, word.xpos))
                # TODO: ignore punctuation?
        # Does not match the paper.
        return dependency_triplets

    def dependency_parse_stanza_initialize(self):
        """Initialization for dependency_parse_stanza()"""
        # nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
        nlp = stanza.Pipeline(lang='en', processors='tokenize,lemma,pos,depparse')
        return nlp

    # padding to the encoding to MAX_VECTOR_LEN
    def padding(self, encoding) -> List[int]:
        PAD_VALUE = 120000   # large meaningless value
        MAX_VECTOR_LEN = 40    # maximum vector length
        fill_vec = MAX_VECTOR_LEN-len(encoding)
        return encoding + [PAD_VALUE] * fill_vec

    # Use this website to create a number for POS, Penn Treebank
    # It seems that punctuation has been removed.
    # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    # This is also here https://catalog.ldc.upenn.edu/docs/LDC95T7/cl93.html in section 2.2, which states "[i]t contains
    # 36 POS tags and 12 other tags (for punctuation and currency symbols)."
    # For my application punctuation could cause issues -> "35,000 sq ft" is a reasonable input.
    UPENN_POS_NUM = {'CC': 1, 'CD': 2, 'DT': 3, 'EX': 4, 'FW': 5, 'IN': 6, 'JJ': 7, 'JJR': 8, 'JJS': 9, 'LS': 10,
                     'MD': 11, 'NN': 12, 'NNS': 13, 'NNP': 14, 'NNPS': 15, 'PDT': 16, 'POS': 17, 'PRP': 18, 'PRP$': 19,
                     'RB': 20, 'RBR': 21, 'RBS': 22, 'RP': 23, 'SYM': 24, 'TO': 25, 'UH': 26, 'VB': 27, 'VBD': 28,
                     'VBG': 29, 'VBN': 30, 'VBP': 31, 'VBZ': 32, 'WDT': 33, 'WP': 34, 'WP$': 35, 'WRB': 36,
                     ### Additional tags
                     # - Don't think hypen should be treated as a POS
                     # Are '-LRB-' and '-RRB-' parenthesis?
                     '.': 37, ',': 38, 'HYPH': 39, '-LRB-': 40, '-RRB-': 41,
                     }

    # Universal Dependency Relations
    # This has 37 universal syntactic relations, but there are also subtype relationships.
    # This totals to 65
    # https://universaldependencies.org/u/dep/index.html
    # no idea if this matches the paper.

    UNIV_DEP_REL_NUM = {
         'aclacl:relcl': 1, 'advcl': 2, 'advmod': 3, 'advmod:emph': 4, 'advmod:lmod': 5, 'amod': 6, 'appos': 7, 'aux': 8,
         'aux:pass': 9, 'case': 10, 'cc': 11, 'cc:preconj': 12, 'ccomp': 13, 'clf': 14, 'compound': 15,
         'compound:lvc': 16, 'compound:prt': 17, 'compound:redup': 18, 'compound:svc': 19, 'conj': 20, 'cop': 21,
         'csubj': 22, 'csubj:outer': 23, 'csubj:pass': 24, 'dep': 25, 'det': 26, 'det:numgov': 27, 'det:nummod': 28,
         'det:poss': 29, 'discourse': 30,  'dislocated': 31, 'expl': 32, 'expl:impers': 33, 'expl:pass': 34,
         'expl:pv': 35, 'fixed': 36, 'flat': 37, 'flat:foreign': 38, 'flat:name': 39, 'goeswith': 40, 'iobj': 41,
         'list': 42, 'mark': 43, 'nmod': 44, 'nmod:poss': 45, 'nmod:tmod': 46,
         'nsubj': 47, 'nsubj:outer': 48, 'nsubj:pass': 49, 'nummod': 50,
         'nummod:gov': 51, 'obj': 52, 'obl': 53, 'obl:agent': 54,
         'obl:arg': 55, 'obl:lmod': 56, 'obl:tmod': 57, 'orphan': 58,
         'parataxis': 59, 'punct': 60, 'reparandum': 61, 'root': 62,
         'vocative': 63, 'xcomp': 64,

         ### additional relationships from spiCy
         'acl': 65,
    }
    # convert dependency triples into numbers.
    #
    # UPENN is the default POS tagger method.
    #
    # the number of UPENN POS tags nltk.help.upenn_tagset(), this prints to console.
    # The paper says the POS tags are 37 in the NLTK library.  I found a few more.  I took out the ones without letters in them.
    # Here's how I filtered those:
    # from nltk.data import load
    # tagdict = load('help/tagsets/upenn_tagset.pickle')
    # tagdict.keys()
    # >>> dict_keys(['LS', 'TO', 'VBN', "''", 'WP', 'UH', 'VBG', 'JJ', 'VBZ', '--', 'VBP', 'NN', 'DT', 'PRP', ':', 'WP$',
    #               'NNPS', 'PRP$', 'WDT', '(', ')', '.', ',', '``', '$', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP', 'JJR',
    #               'JJS', 'PDT', 'MD', 'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS'])
    #
    #  len(tagdict.keys())
    # >>> 45
    # filt=list(filter(lambda x: x.isalpha(), tagdict.keys()))
    # len(filt)
    # >>> 34
    # Perhaps the actual source is this website which lists 36 POS:
    # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

    # There's not enough information here about how this was converted.  I think it is just best to be able to reverse
    # the process.  Could keep a list of used dependency triples, if keeping the numbers smaller might be helpful.
    # Smaller numbers are more frequent.

    # This was helpful: https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk
    def label_encoding(self, dep_triples: list):
        # work on a single triple
        def label_encode(triple: tuple):
            NUM_DEPS = 66
            # NUM_POS = 37  # paper
            NUM_POS = 42
            # triple[0] encode depedencies
            return (self.UPENN_POS_NUM[triple[0]] * NUM_DEPS + self.UNIV_DEP_REL_NUM[triple[1]]) * NUM_POS + self.UPENN_POS_NUM[triple[2]]

 #       return map(label_encode, dep_triples)
        return [label_encode(t) for t in dep_triples]

    # NOT WORKING, probably should DELETE
    # 63 dependency tags listed in the paper is close to 65 listed in the Universal Dependency Relations
    # Perhaps a couple are removed for not being needed.
    # https://universaldependencies.org/u/dep/


    # This sets up everything needed to determine the maximum vector size.
    # This only needs to be performed once
    # This takes about 3 minutes to run on CPU.
    def run_compute_max_length_from_training_data(self):
        # tg = generate_templates()
        tmpls = self.tg.generate_all_templates(self.kg, self.kg)
        maximum = self.compute_max_length_from_training_data([generated_data['question'] for generated_data in tmpls])
        return maximum

    # This is really just something that needs to be ran once to determine the size of l1 and l2
    # I used this to inform the MAX_VECTOR_LEN in padding().
    def compute_max_length_from_training_data(self, training_data):
        # nlp = self.dependency_parse_stanza_initialize()
        max_sentence = None
        max_label_encoding = []
        for sentence in training_data:
            if '-' in sentence:
                print(f"SENTENCE as a hypen: {sentence}")
            dependency_tripe = self.dependency_parse_stanza(self.nlp, sentence)
            encoding = list(self.label_encoding(dependency_tripe))
            if len(encoding) > len(max_label_encoding):
                max_label_encoding = encoding
                max_sentence = sentence

        return max_sentence, max_label_encoding

    # do all the dependency parsing in one go and free all the memory of this.
    def dependency_encoding(self, nlp, sentence: str) -> list:
        tpl = self.dependency_parse_stanza(nlp, sentence)
        enc = self.label_encoding(tpl)
        return self.padding(enc)

    # WORKING
    def train2(self, question_templates: Optional[list] = None):
        """Train the Question XGBoost Classifier.
        train_iter : iterable of the questions that you'd like to train"""
        logger.info("1. Question Classifier training begun. ================")
        start_time = time.time()
        # generate the questions from the templates
        if question_templates is not None:
            gen_temp = question_templates
        else:
            # use the default
            gen_temp = self.tg.generate_all_templates_shuffle(self.kg, self.kg)

        # NOTE: the steps are is specific to this dataset.
        questions_and_labels = [(gd['question'], gd['template_name']) for gd in gen_temp]
        question_corpus = [ql[0] for ql in questions_and_labels]
        print(f"question_corpus: {question_corpus}")
#        template_number_dict = label_dictionary()
#        template_labels = [template_number_dictionary()[ql[1]] for ql in questions_and_labels]
        template_labels = [self.tg.template_number_dict[ql[1]] for ql in questions_and_labels]



#        question_dep_encoded = np.array(map(lambda sentence: np.array(dependency_steps(nlp, sentence)), question_corpus))
        question_dep_encoded = np.array([np.array(self.dependency_encoding(self.nlp, sentence)) for sentence in question_corpus])
        print(f"Number of Questions Encoded: {question_dep_encoded.shape[0]}")
        print(template_labels)
        print(question_dep_encoded)
#        X_train, X_test, y_train, y_test = train_test_split(question_dep_encoded, template_labels) # default test_size=0.25
#        X_train, X_test, y_train, y_test = train_test_split(question_dep_encoded, template_labels, test_size=0.2, random_state=246341428)
        X_train, X_test, y_train, y_test = train_test_split(question_dep_encoded, template_labels,
                                                            test_size=0.25, random_state=246341428, shuffle=True)

        param = {'booster': 'gbtree',
                 'learning_rate': 0.3,
                 'max_depth': 3,
                 'subsample': 0.5,
                 'colsample_bytree': 0.4,
                 'objective': 'multi:softprob',
                 'sample_type': 'uniform',
                 'tree_method': 'auto',
         }
        model = xgb.XGBClassifier()  # 98.12% accuracy without parameters
#         model = xgb.XGBClassifier(*param)  # 96.99% accuracy with parameters
        model.fit(X_train, y_train)

        # make predictions for test data
        print(f'X_test {X_test}')
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='micro')
        print("Accuracy %.2f%%" % (accuracy * 100.0))
        print(f"F1 Score: {f1* 100.0}")
        model.save_model("question_classification_model.ubj")
        end_time = time.time()
        logger.info(f"Training XGBoost Classifier took: {end_time-start_time} s")
        self.model = model
        return model


    def classify(self, sentence: str, model=None) -> int:
        """Classify a question based on the sentence string.
        model would be used for classifying multiple sentences or a long running process.
        returns an integer for the question classification

        sentence - sentence (string) to be classified
        model - XGBoost Classifier model

        returns an integer with the classification"""

        question_dep_encoded = np.array([self.dependency_encoding(self.nlp, sentence)])
        # using self.model instead
#        if model is None:
#            model = self.load_model()
#            logger.info("Loading XGBoost model: question_classification_model.ubj")
#            model = xgb.XGBClassifier()
#            model.load_model("question_classification_model.ubj")

        print(question_dep_encoded)
        ypred = self.model.predict(question_dep_encoded)
        print(ypred)
        return ypred[0]

    def classification_number_to_template_name(self, number: int) -> str:
        """Convert the number of the template to a name"""
#        return template_names()[number]
        return self.tg.template_names()[number]

    def load_model(self):
        logger.info("Loading XGBoost model: question_classification_model.ubj")
        model = xgb.XGBClassifier()
        # TODO: If this gives an error, need to suggest training first.
        model.load_model("question_classification_model.ubj")
        return model


### These are main functions to make stuff happen.

def main():
    """example to replicate paper"""
    qc = QuestionClassification()
    qc.run("What is the address of the hotel where Mozart Week takes place?")
#    qc.run("When does Mozart Week start?")

# this function seems to work.  - takes 16 minutes on newer set of questions.
def qc_train_main():
    """This is a step to train the classifier"""
    qc = QuestionClassification()
    qc.train2()

def compute_max_main():
    """Exercise to figure out the maximum vector needed for the problem."""
    qc = QuestionClassification()
    maximum = qc.run_compute_max_length_from_training_data()
    print(f'MAXIMUM: {maximum}')
    print(f'length: {len(maximum)}')

# this is a test to make sure that this function can get to the TTL files.
# def generate_all_templates_text():
#    for res in generate_templates():
#        print(res)

def classify_small_test_main():
    qc = QuestionClassification()
#    model = qc.train2()
#    template_number = qc.classify('Are auto-dismantling yards permitted?', model)

    # these are questions to test that all of the 5 of the templates are being tested
    questions = ['Are auto-dismantling yards permitted?',
                 'Is the maximum building height for a property in the C3 zoning district 50 feet?',
                 'Which zoning districts allow physical fitness centers?',
                 'Are salt works allowed in a FI3 zoning district?',
                 'What is the maximum building height in the R2 zoning district?',]

    print("===== Question Classifications =====")
    model = qc.load_model()
    for q in questions:
        template_number = qc.classify(q, model)
        print(f"Question: {q} template_number = {template_number}")

# On CPU this takes about 4 minutes
def classify_all_test_main():
    qc = QuestionClassification()
    print("===== Question Classifications =====")
    model = qc.load_model()
    tmpls = self.tg.generate_all_templates(kg, kg)
    questions = [generated_data['question'] for generated_data in tmpls]
#    for q in questions:
#        template_number = qc.classify(q, model)
    [qc.classify(q, model) for q in questions]




if __name__ == '__main__':
#    sys.exit(main())
#    sys.exit(compute_max_main())
#    sys.exit(generate_all_templates_text())

    sys.exit(qc_train_main())
#    classify_small_test_main()
#    classify_all_test_main()
