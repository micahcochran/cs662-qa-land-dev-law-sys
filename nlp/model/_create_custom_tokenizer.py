""" _summary_
"""

from ..corpus import Corpus

import os
import shutil
from pathlib import Path
from nltk import sent_tokenize
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizer
from dataclasses import dataclass

@dataclass
class CustomTokenizer:
    """ _summary_

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    custom_tokenizer: type=None
    subword_tokenizer: type=None
    temp_path: str=None
    model_path: str=None
    name: str=None
    
    def _train_tokenizer(self, tokenizer, vocab_size: int, special_tokens: list, model_dir: str) -> type:
        """_train_tokenizer _summary_

        Args:
            tokenizer (_type_): _description_
            vocab_size (int): _description_
            special_tokens (list): _description_
            name (str): _description_
            model_dir (str): _description_

        Returns:
            type: _description_
        """
        self.model_path = model_dir
        paths = [str(x) for x in Path(self.temp_path).glob('**/*.txt')]
        self.subword_tokenizer = tokenizer
        self.subword_tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=1, special_tokens=special_tokens)
        self.subword_tokenizer.save_model(self.model_path)
        return self.subword_tokenizer
    
    def _create_batches(self, corpus: Corpus, temp_dir: str) -> str:
        """_create_batches _summary_

        Args:
            corpus (Corpus): _description_
            temp_dir (str): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            str: _description_
        """
        text_data = []
        file_count = 0
        self.name = corpus.name
        self.temp_path = temp_dir
        for text in corpus.raw_texts:
            for string in sent_tokenize(text):
                text_data.append(string)
                if len(text_data) == 10_000:
                    with open(f'{temp_dir}/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
                        try:
                            fp.write('\n'.join(text_data))
                        except:
                            raise ValueError(f"Invalid file: {fp}")
                    text_data = []
                    file_count += 1
        with open(f'{temp_dir}/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
            try:
                fp.write('\n'.join(text_data))
            except:
                raise ValueError(f"Invalid file: {fp}")
        return self.temp_path
    
    def _initialize_custom_tokenizer(self, transform: type) -> type:
        self.custom_tokenizer = transform.from_pretrained(self.model_path, max_len=512)
        return self.custom_tokenizer
        
def create_custom_tokenizer(corpus: Corpus, temp_dir: str, model_dir: str, transform: type=RobertaTokenizer, tokenizer: type=ByteLevelBPETokenizer, special_tokens: list=['<s>', '</s>', '<pad>', '<unk>', '<mask>']) -> CustomTokenizer:
    """create_custom_tokenizer _summary_

    Args:
        corpus (Corpus): _description_
        temp_dir (str): _description_
        model_dir (str): _description_
        transform (type, optional): _description_. Defaults to type[RobertaTokenizer].
        tokenizer (type, optional): _description_. Defaults to type[ByteLevelBPETokenizer].
        special_tokens (list, optional): _description_. Defaults to ['<s>', '</s>', '<pad>', '<unk>', '<mask>'].

    Returns:
        CustomTokenizer: _description_
    """
    token = CustomTokenizer()
    token._create_batches(corpus, temp_dir)
    token._train_tokenizer(tokenizer=tokenizer, vocab_size=len(corpus.vocabulary), special_tokens=special_tokens, model_dir=model_dir)
    token._initialize_custom_tokenizer(transform=transform)
    return token
