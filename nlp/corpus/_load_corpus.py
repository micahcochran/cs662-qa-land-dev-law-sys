""" _summary_
"""

import ntpath
import numpy as np
import pandas as pd 
from typing import Callable, Tuple
from dataclasses import dataclass, field
from importlib import resources


DATA_MODULE = "nlp.corpus.data"

@dataclass
class Corpus:
    
    corpus: dict=field(default_factory=dict)
    name: str=None
    file_folder: str=None
    raw_text: list=field(default_factory=list)
    sents: list=field(default_factory=list)
    words: list=field(default_factory=list)
    tags: list=field(default_factory=list)
    
    
    def _load_text(self, data_module=DATA_MODULE, file_folder: str=None) -> list:
        data_module = f"{data_module}.{file_folder}"
        self.name = file_folder
        try:
            files = resources.files(data_module).iterdir()
        except:
            raise ValueError(f"Invalid file folder: {file_folder}")
        for file in files:
            file = ntpath.basename(file)
            if str(file) not in ["__init__.py", "__pycache__"] and ntpath.isfile(file) == False:
                try:
                    text = resources.read_text(data_module, file)
                    self.raw_text.append(text)
                except:
                    raise ValueError(f"Invalid file: {file}")
        self.corpus['texts'] = self.raw_text
        return self.raw_text
        
    def _load_sents(self) -> list:
        return self.sents
    
    def _load_words(self) -> list:
        return self.words
    
    def _load_tags(self) -> list:
        return self.tags

def load_corpus(suffix: str) -> Corpus:
    """load_corpus _summary_

    Returns:
        Corpus: _description_
    """
    corp = Corpus()
    corp._load_text(file_folder=suffix)
    return corp