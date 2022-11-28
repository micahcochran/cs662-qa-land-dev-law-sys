""" _summary_
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import json
import ntpath
import datasets
from importlib import resources
from dataclasses import dataclass, field

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification

from haystack.nodes import FARMReader
from haystack.nodes import TfidfRetriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import clean_wiki_text, convert_files_to_docs

DATA_MODULE = "nlp.model.data.question_answering"

@dataclass
class SequenceClassificationBasedModel():
    name: str=None
    data: datasets.Dataset = None
    model: AutoModelForSequenceClassification = None
    model_dir: str=None
    
    def _preprocess_function(examples, tokenizer):
        return tokenizer(examples["text"], truncation=True)
    
    def _create_model(self, pretrained: str, model_dir: str, gpu: bool, data: datasets.Dataset, n_classes: int):
        self.model = None 
        return self.model

def create_sequence_classification_model():
    return None

@dataclass
class SquadBasedModel():

    reader: FARMReader=None
    document_store: InMemoryDocumentStore=None
    retriever: TfidfRetriever=None
    model: ExtractiveQAPipeline=None
    model_dir: str=None
    
    def _create_model(self, reader: type, doc_dir: str) -> ExtractiveQAPipeline:
        self.document_store = InMemoryDocumentStore()
        docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
        self.document_store.write_documents(docs)
        self.retriever = TfidfRetriever(document_store=self.document_store)
        self.reader = reader
        self.moel = ExtractiveQAPipeline(self.reader, self.retriever)
        return self.model
    
def create_squad_model(reader: type=FARMReader, doc_dir: str= f"{os.getcwd()}") -> SquadBasedModel:
    model = SquadBasedModel(reader=reader, doc_dir=doc_dir)
    return model
    
@dataclass
class CustomReader():
    """CustomReader _summary_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    name: str=None
    data: json=field(default_factory=dict)
    reader: FARMReader = None
    model_dir: str=None
    
    def _create_reader(self, data_name: str, pretrained: str, gpu: bool, epochs: int, model_dir: str, dev_split: float, max_seq_len: int, data_module=DATA_MODULE) -> FARMReader:
        """_create_reader _summary_

        Args:
            data_name (str): _description_
            pretrained (str): _description_
            gpu (bool): _description_
            model_dir (str): _description_
            data_module (_type_, optional): _description_. Defaults to DATA_MODULE.

        Raises:
            ValueError: _description_

        Returns:
            FARMReader: _description_
        """
        data_module = f'{data_module}.{data_name}'
        self.name = data_name
        try:
            files = resources.files(data_module).iterdir()
        except:
            raise  ValueError(f"Invalid file folder: {data_name}")
        for file in files:
            print(file)
            if ".json" in str(file):
                with open(file) as json_file:
                    self.data = json.load(json_file)
                path_split = ntpath.split(file)
        self.reader = FARMReader(model_name_or_path=pretrained, use_gpu=gpu)
        self.reader.train(data_dir=path_split[0], train_filename=path_split[1], dev_split=dev_split, max_seq_len=max_seq_len, use_gpu=gpu, n_epochs=epochs, save_dir="my_model")
        self.model_dir = model_dir
        self.reader.save(model_dir)
        return self.reader
        
def create_reader(model_dir: str, data_name: str, dev_split: float=0.1, max_seq_len: int=256, pretrained: str="deepset/roberta-base-squad2", gpu: bool=False, epochs:int=1) -> FARMReader:
    """create_reader _summary_

    Args:
        model_dir (str): _description_
        data_name (str): _description_
        pretrained (str, optional): _description_. Defaults to "deepset/roberta-base-squad2".
        gpu (bool, optional): _description_. Defaults to False.

    Returns:
        FARMReader: _description_
    """
    reader = CustomReader()
    reader._create_reader(data_name=data_name, pretrained=pretrained, gpu=gpu, model_dir=model_dir, epochs=epochs, dev_split=dev_split, max_seq_len=max_seq_len)
    return reader