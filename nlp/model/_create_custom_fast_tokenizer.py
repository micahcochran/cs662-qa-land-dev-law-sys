""" _summary_
"""

from ..corpus import Corpus

from pathlib import Path
from nltk import sent_tokenize

from transformers import RobertaTokenizerFast
from dataclasses import dataclass

@dataclass
class CustomFastTokenizer:
    custom_tokenizer: type=None
    pretrained_tokenizer: type=None
    temp_path: str=None
    model_path: str=None
    name: str=None
    
    def _train_tokenizer(self, name: str, transform: type, raw_texts: list, vocab_size: int, model_dir: str, pretrained_model: str, batch_size: int) -> type:
        self.name = name
        sents = []
        [[sents.append(sent) for sent in sent_tokenize(text)] for text in raw_texts]
        self.pretrained_tokenizer = transform.from_pretrained(pretrained_model)
        self.custom_tokenizer = self.pretrained_tokenizer.train_new_from_iterator(text_iterator=self._batch_iterator(sents, batch_size), vocab_size=vocab_size)
        self.custom_tokenizer.save_pretrained(model_dir)
        return self.custom_tokenizer
    
    def _batch_iterator(self, raw_data, batch_size: int=10000):
        for i in range(0, len(raw_data), batch_size):
            yield raw_data[i : i + batch_size]
            
def create_custom_fast_tokenizer(corpus: Corpus, model_dir: str, transform: type=RobertaTokenizerFast, batch_size: int=10000) -> CustomFastTokenizer:
    raw_texts = corpus.raw_texts
    name =  f'{corpus.name}_fast_tokenizer'
    custom = CustomFastTokenizer()
    custom._train_tokenizer(name=name, transform=transform, raw_texts=raw_texts, vocab_size=len(corpus.vocabulary), model_dir=model_dir, pretrained_model='KoichiYasuoka/roberta-base-english-upos', batch_size=batch_size) 
    return custom