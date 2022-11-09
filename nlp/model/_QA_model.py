""" _summary_
"""

import json
import ntpath
from importlib import resources
from dataclasses import dataclass, field
import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from haystack.nodes import FARMReader

DATA_MODULE = "nlp.model.data.question_answering"

@dataclass
class SquadBasedModel():
    
    name: str=None
    
    def _create_model(self, reader: type) -> list:
        return None
    
@dataclass
class CustomReader():
    
    name: str=None
    data: json=field(default_factory=dict)
    reader: FARMReader = None
    model_dir: str=None
    
    def _create_reader(self, data_name: str, pretrained: str, gpu: bool, model_dir: str, data_module=DATA_MODULE) -> FARMReader:
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
        self.reader.train(data_dir=path_split[0], train_filename=path_split[1], use_gpu=True, n_epochs=1, save_dir="my_model")
        self.model_dir = model_dir
        self.reader.save(model_dir)
        return self.reader
        
def create_reader(model_dir: str, data_name: str, pretrained: str="deepset/roberta-base-squad2", gpu: bool=False) -> FARMReader:
    reader = CustomReader()
    reader._create_reader(data_name=data_name, pretrained=pretrained, gpu=gpu, model_dir=model_dir)
    return reader