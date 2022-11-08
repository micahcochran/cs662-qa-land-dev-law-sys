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
    
    def _train_tokenizer(self, tokenizer, raw_text: str, vocab_size: int, model_dir: str, pretrained_model: str, batch_size: int) -> type:
        self.pretrained_tokenizer = tokenizer.from_pretrained(pretrained_model)
        self.custom_tokenizer = self.pretrained_tokenizer.train_new_from_iterator(text_iterator=self._batch_iterator(raw_text, batch_size), vocab_size=vocab_size)
        self.custom_tokenizer.save_pretrained(model_dir)
        return None
    
    def _batch_iterator(self, raw_data, batch_size: int=10000):
        for i in range(0, len(raw_data), batch_size):
            yield raw_data[i : i + batch_size]["text"]
            
def create_custom_fast_tokenizer(corpus: Corpus, temp_dir: str, model_dir: str, transform: type=RobertaTokenizerFast) -> CustomFastTokenizer:
    return None