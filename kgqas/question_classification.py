"""
Semantic Parsing Phase - 1) Question Classification
"""

# training takes ~3 minutes on CPU

import itertools
# Python library imports
import sys
import time
from typing import Generator, List, Tuple

# External library imports
# install loguru
from loguru import logger
from nltk.tag.perceptron import PerceptronTagger
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
import spacy
import stanza
# install xgboost
import xgboost as xgb

# The below aren't needed for nltk because I am using spacy.
# Run these one time only
import nltk
# nltk.download('universal_tagset')    # brown when tagset='universal'
# nltk.download('averaged_perceptron_tagger')    # PerceptronTagger(load=True)

class QuestionClassification:
    def pos_tagging(self, sentence: str):
        avg_tagger = PerceptronTagger(load=True)
        tokens = word_tokenize(sentence)
        avg_perceptron_tags = avg_tagger.tag(tokens)
        return avg_perceptron_tags

    # result in a triplets in the following form:
    # <POS tag of headword, dependency tag, POS tag of dependent word>
    def dependency_parsing_spacy(self, sentence: str) -> List[tuple]:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentence)
        # see, https://spacy.io/api/token for explanation
        for token in doc:
#            print(type(token))
#            print(
#                f'{token.text:{12}} {token.pos_:{6}} {token.tag_:{6}} {token.dep_:{6}} {spacy.explain(token.pos_):{20}} {spacy.explain(token.tag_)}')
            print(
               f'{token.text:{12}} {token.pos_:{6}} <{token.tag_:{6}} {token.dep_:{6}}> {token.head} {spacy.explain(token.pos_):{20}} {spacy.explain(token.tag_)}')
        return [(doc[token.i].tag_, token.head, token.dep_, token.tag_) for token in doc]

    # result in a triplets in the following form:
    # <POS tag of headword, dependency tag, POS tag of dependent word>
    # This WORKS
    # This uses a Standford parser.
    def dependency_parse_stanza(self, nlp, sentence: str) -> List[tuple]:
        doc = nlp(sentence)
#        print(
#            *[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for
#              sent in doc.sentences for word in sent.words], sep='\n')
#        print(*[
#            f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head - 1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}'
#            for sent in doc.sentences for word in sent.words], sep='\n')
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
        nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
        return nlp

    # padding to the encoding to MAX_VECTOR_LEN
    def padding(self, encoding):
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
#            print(triple)
#            print(self.UPENN_POS_NUM[triple[0]])
            return (self.UPENN_POS_NUM[triple[0]] * NUM_DEPS + self.UNIV_DEP_REL_NUM[triple[1]]) * NUM_POS + self.UPENN_POS_NUM[triple[2]]

 #       return map(label_encode, dep_triples)
        return [label_encode(t) for t in dep_triples]

    # NOT WORKING, porbably should DELETE
    # 63 dependency tags listed in the paper is close to 65 listed in the Universal Dependency Relations
    # Perhaps a couple are removed for not being needed.
    # https://universaldependencies.org/u/dep/
    def classifier(self, train=False):
        # xgboost goes here
        # parameters the specified in Table 4 of the paper
        param = {'booster': 'gbtree',
                 'learning_rate': 0.3,
                 'max_depth': 3,
                 'subsample': 0.5,
                 'colsample_bytree': 0.4,
                 'objective': 'multi:softprob',
                 'sample_type': 'uniform',
                 'tree_method': 'auto',
                 }
        # xgb.train(param)
        # create model instance
        #bst = XGBClassifier(kwargs=param)
        # if train is True:
        #    xgb.train(param, )
        # fit model
        # bst.fit(X_train, y_train)
        # make predictions
        # else:
        preds = bst.predict(X_test)

    def generate_templates(self) -> itertools.chain[dict]:
        """build the templates"""
        sys.path.append("..")  # hackish
        from triplesdb.generate_template import generate_all_templates
        import rdflib
        uses_kg = rdflib.Graph()
        uses_kg.parse("../triplesdb/combined.ttl")

        dimreq_kg = rdflib.Graph()
        # one of the templates is unable handle combined.ttl
        dimreq_kg.parse("../triplesdb/bulk2.ttl")

        tg_iter = generate_all_templates(uses_kg, dimreq_kg)

        return tg_iter

    def label_dictionary(self) -> dict:
        sys.path.append("..")  # hackish
        from triplesdb.generate_template import TemplateGeneration
        tg_obj = TemplateGeneration()
        return tg_obj.template_number_dict

    # This sets up everything needed to determine the maximum vector size.
    # This takes about 3 minutes to run on CPU.
    def run_compute_max_length_from_training_data(self):
        tg = self.generate_templates()
        maximum = self.compute_max_length_from_training_data([generated_data['question'] for generated_data in tg])
        return maximum

    # This is really just something that needs to be ran once to determine the size of l1 and l2
    # I used this to inform the MAX_VECTOR_LEN in padding().
    def compute_max_length_from_training_data(self, training_data):
        nlp = self.dependency_parse_stanza_initialize()
        max_sentence = None
        max_label_encoding = []
        for sentence in training_data:
            if '-' in sentence:
                print(f"SENTENCE as a hypen: {sentence}")
            dependency_tripe = self.dependency_parse_stanza(nlp, sentence)
            encoding = list(self.label_encoding(dependency_tripe))
            if len(encoding) > len(max_label_encoding):
                max_label_encoding = encoding
                max_sentence = sentence

        return max_sentence, max_label_encoding

    # WORKING
    def train2(self):
        logger.info("1. Question Classifier training begun. ================")
        start_time = time.time()
        # generate the questions from the templates
        tg = self.generate_templates()

        questions_and_labels = [(gd['question'], gd['template_name']) for gd in tg]
        question_corpus = [ql[0] for ql in questions_and_labels]
        template_number_dict = self.label_dictionary()
        template_labels = [template_number_dict[ql[1]] for ql in questions_and_labels]

        # note subsetting the data set barely helped the speed
#        SUBSET = 100
        SUBSET = 0
        # create a subset of the data  (SUBSET=0 to disable creating a subset)
        if SUBSET > 0:
            question_corpus = question_corpus[:SUBSET]
            template_labels = np.array(list(template_labels)[:SUBSET])

        # do all the dependency parsing in one go and free all the memory of this.
        def dependency_steps(nlp, sentence):
            tpl = self.dependency_parse_stanza(nlp, sentence)
            enc = self.label_encoding(tpl)
            return self.padding(enc)


        nlp = self.dependency_parse_stanza_initialize()

#        question_dep_encoded = np.array(map(lambda sentence: np.array(dependency_steps(nlp, sentence)), question_corpus))
        question_dep_encoded = np.array([np.array(dependency_steps(nlp, sentence)) for sentence in question_corpus])
        print(f"Number of Questions Encoded: {question_dep_encoded.shape[0]}")
        print(template_labels)
        print(question_dep_encoded)
        X_train, X_test, y_train, y_test = train_test_split(question_dep_encoded, template_labels, shuffle=True, random_state=42)
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
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='macro')
        print("Accuracy %.2f%%" % (accuracy * 100.0))
        print(f"F1 Score: {f1* 100.0}")
        model.save_model("question_classification_model.ubj")

    # NOT WORKING
    def train(self):
        """Training for the Question Classification"""
        logger.info("1. Question Classifier training begun. ================")
        start_time = time.time()
        # generate the questions from the templates
        tg = self.generate_templates()

        questions_and_labels = [(gd['question'], gd['template_name']) for gd in tg]
        question_corpus = [ql[0] for ql in questions_and_labels]
#        question_corpus = map(lambda gd: gd['question'], tg)
        # numeric template labels (a parallel array to the question_corpus)
#        template_labels = map(lambda gd: np.array([tg.template_number_dict[gd['template_name']]]), tg)
#        template_labels = map(lambda gd: tg.template_number_dict[gd['template_name']], tg)
 #       template_labels = [tg.template_number_dict[gd['template_name']] for gd in tg.template]
        template_number_dict = self.label_dictionary()
#        template_labels_int = [ql[1] for ql in questions_and_labels]
#        template_labels = template_number_dict[
        template_labels = [template_number_dict[ql[1]] for ql in questions_and_labels]
#        print([gd['template_name'] for gd in tg])

        SHORT_TEST = True
        # create a subset of the data
        if SHORT_TEST:
            SUBSET = 100
            question_corpus = question_corpus[:SUBSET]
            template_labels = np.array(list(template_labels)[:SUBSET])


        logger.info("Dependency & POS tagging")
        nlp = self.dependency_parse_stanza_initialize()
        # generate the dependency triples
        dep_triples = [self.dependency_parse_stanza(nlp, sentence) for sentence in question_corpus]
#        dep_triples = map(lambda s: self.dependency_parse_stanza(nlp, s), question_corpus)

        logger.info("Dependency & POS tagging - Done")

        logger.info("Encode Dependency Triples")
        # encode the dependency triples
        enc_dep_vec = [self.label_encoding(tpl) for tpl in dep_triples]
        logger.info("Encode Dependency Triples - Done")


        logger.info("Padding List")
#        enc_dep_vec = map(lambda tpl: self.label_encoding(tpl), dep_triples)
        # pad
        pad_enc_dep_vec = [self.padding(enc) for enc in enc_dep_vec]
#        pad_enc_dep_np = [np.array(self.padding(enc)) for enc in enc_dep_vec]
        logger.info("Padding List - Done")

        # import pickle
        # pickle.dump(pad_enc_dep_vec, "pad_enc_dep_vec.pickle")
        # pickle.dump(template_labels, "template_labels.pickle")
#        print(np.array(pad_enc_dep_vec))
#        print(np.array(template_labels))
#        return
        self.train_xgb(np.array(pad_enc_dep_vec), np.array(template_labels))
        return
#        logger.info("Convert to DMatrix")
        # Is this right?????????????????????????????????????????????????????????????
        #        pad_enc_dep_dmatrixes = [xgb.DMatrix(tpl) for tpl in enc_dep_triples]
        data_dmatrix = xgb.DMatrix(pad_enc_dep_vec, template_labels)
#         template_label_dmatrix = xgb.DMatrix()
#        template_label_dmatrix = xgb.DMatrix([list(x) for x in template_labels])

#        logger.info("Convert to DMatrix - Done")

        # parameters the specified in Table 4 of the paper
        param = {'booster': 'gbtree',
                 'learning_rate': 0.3,
                 'max_depth': 3,
                 'subsample': 0.5,
                 'colsample_bytree': 0.4,
                 'objective': 'multi:softprob',
                 'sample_type': 'uniform',
                 'tree_method': 'auto',
                 }

        # train using 5 fold cross validation --- not needed for XGBoost
        kf = KFold(n_splits=5)
        for testing_data, training_data in kf.split(dep_triples):
#        for testing_data, training_data in self.cross_validation_lists(dep_triples):
            xgb.train(param, dtrain=training_data)

        print(data_dmatrix)
        # train using 5 fold cross validation
#        eval_hist = xgb.cv(param, data_dmatrix, nfold=5)
        print(f"eval_hist: {eval_hist}")
        # save the trained model
        bst.save_model("question_classification.ubj")
        # fit model
        # bst.fit(X_train, y_train)
        preds = bst.predict(pad_enc_dep_dmatrixes)
        end_time = time.time()
        logger.info(f"QC's train function took {end_time-start_time} s")

    # NOT WORKING
    # I used this example for guidance
    def train_xgb(self, X, Y):
        param = {'booster': 'gbtree',
                 'learning_rate': 0.3,
                 'max_depth': 3,
                 'subsample': 0.5,
                 'colsample_bytree': 0.4,
                 'objective': 'multi:softprob',
                 'sample_type': 'uniform',
                 'tree_method': 'auto',
                 }
        # split data into train and test sets
       # seed = 7
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size) # , random_state=seed)
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))


    def run(self, sentence: str):
        # pos = self.pos_tagging(sentence)
        # logger.info(f"POS Tagging {pos}")
#       dep_triples = self.dependency_parsing_spacy(sentence)
        dep_triples = self.dependency_parse_stanza(sentence)
        logger.info(f"Dependency_triples:  {dep_triples}")



### These are main functions to make stuff happen.

def main():
    qc = QuestionClassification()
    qc.run("What is the address of the hotel where Mozart Week takes place?")
#    qc.run("When does Mozart Week start?")

# this function seems to work.
def qc_train_main():
    """This is a step to train the classifier"""
    qc = QuestionClassification()
#    qc.train()
    qc.train2()

def compute_max_main():
    """Exercise to figure out the maximum vector needed for the problem."""
    qc = QuestionClassification()
    maximum = qc.run_compute_max_length_from_training_data()
    print(f'MAXIMUM: {maximum}')
    print(f'length: {len(maximum)}')

def generate_all_templates_text():
    qc = QuestionClassification()
    for res in qc.generate_templates():
        print(res)

if __name__ == '__main__':
#    sys.exit(main())
#    sys.exit(compute_max_main())
    sys.exit(qc_train_main())
#    sys.exit(generate_all_templates_text())