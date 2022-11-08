""" _summary_
"""

import nltk
import ntpath
from importlib import resources
from nltk.tokenize import word_tokenize, sent_tokenize
from dataclasses import dataclass, field

DATA_MODULE = "nlp.corpus.data"

@dataclass
class Corpus:
    """ _summary_

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    corpus: dict=field(default_factory=dict)
    name: str=None
    raw_texts: list=field(default_factory=list)
    sents: list=field(default_factory=list)
    words: list=field(default_factory=list)
    vocabulary: list=field(default_factory=list)
    tagged_words: list=field(default_factory=list)
    tagged_sents: list=field(default_factory=list)
    
    
    def _load_text(self, data_module=DATA_MODULE, corpus_name: str=None) -> list:
        """_load_text _summary_

        Args:
            data_module (_type_, optional): _description_. Defaults to DATA_MODULE.
            corpus_name (str, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            list: _description_
        """
        data_module = f"{data_module}.{corpus_name}"
        self.name = corpus_name
        try:
            files = resources.files(data_module).iterdir()
        except:
            raise ValueError(f"Invalid file folder: {corpus_name}")
        for file in files:
            file = ntpath.basename(file)
            if str(file) not in ["__init__.py", "__pycache__"] and ntpath.isfile(file) == False:
                try:
                    text = resources.read_text(data_module, file)
                    text = ' '.join(text.splitlines())
                    self.raw_texts.append(text)
                except:
                    raise ValueError(f"Invalid file: {file}")
        self.corpus['texts'] = self.raw_texts
        return self.raw_texts
        
    def _load_sents(self, language: str) -> list:
        """_load_sents _summary_

        Args:
            language (str): _description_

        Raises:
            ValueError: _description_

        Returns:
            list: _description_
        """
        nltk.download('punkt')
        for text in self.raw_texts:
            try:
                sentence = [word_tokenize(sent, language=language) for sent in sent_tokenize(text, language=language)]
                for sent in sentence:
                    if sentence:
                        self.sents.append(sent)
            except:
                raise ValueError(f"Invalid sentence: {sent}")
        self.corpus['sents'] = self.sents
        return self.sents
    
    def _load_words(self) -> list:
        """_load_words _summary_

        Returns:
            list: _description_
        """
        for sent in self.sents:
            [self.words.append(word) for word in sent]
        self.corpus['words'] = self.words
        return self.words
    
    def _load_vocabulary(self) -> list:
        """_load_vocabulary _summary_

        Returns:
            list: _description_
        """
        vocab = list(set(self.words))
        self.vocabulary = vocab
        self.corpus['vocabulary'] = self.vocabulary
        return self.vocabulary
        
    def _load_tagged_sents(self, tagset: str) -> list:
        """_load_tagged_sents _summary_

        Args:
            tagset (str): _description_

        Raises:
            ValueError: _description_

        Returns:
            list: _description_
        """
        nltk.download(tagset)
        nltk.download('averaged_perceptron_tagger')
        try:
            if 'universal' in tagset:
                self.tagged_sents = nltk.pos_tag_sents(self.sents, tagset='universal')
            else:
                self.tagged_sents = nltk.pos_tag_sents(self.sents)
        except:
            raise ValueError(f"Invalid sentence set")
        self.corpus['tagged_sents'] = self.tagged_sents
        return self.tagged_sents
    
    def _load_tagged_words(self) -> list:
        """_load_tagged_words _summary_

        Raises:
            ValueError: _description_

        Returns:
            list: _description_
        """
        for sent in self.tagged_sents:
            [self.tagged_words.append(tagged_word) for tagged_word in sent]
        self.corpus['tagged_words'] = self.tagged_words
        return self.tagged_words

def load_corpus(corpus_name: str, language: str='english', tagset: str='universal_tagset') -> Corpus:
    """load_corpus _summary_

    Returns:
        Corpus: _description_
    """
    corp = Corpus()
    corp._load_text(corpus_name=corpus_name)
    corp._load_sents(language)
    corp._load_words()
    corp._load_vocabulary()
    corp._load_tagged_sents(tagset)
    corp._load_tagged_words()
    return corp
