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
class SequenceClassificationBasedModel():
    
    def _create_model():
        return None

@dataclass
class SquadBasedModel():
    
    name: str=None
    
    def _create_model(self, reader: type) -> list:
        return None
    
def create_squad_model() -> SquadBasedModel:
    return SquadBasedModel()
    
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
    
    def _create_reader(self, data_name: str, pretrained: str, gpu: bool, epochs: int, model_dir: str, data_module=DATA_MODULE) -> FARMReader:
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
        self.reader.train(data_dir=path_split[0], train_filename=path_split[1], use_gpu=gpu, n_epochs=1, save_dir="my_model")
        self.model_dir = model_dir
        self.reader.save(model_dir)
        return self.reader
        
def create_reader(model_dir: str, data_name: str, pretrained: str="deepset/roberta-base-squad2", gpu: bool=False, epochs:int=1) -> FARMReader:
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
    reader._create_reader(data_name=data_name, pretrained=pretrained, gpu=gpu, model_dir=model_dir, epochs=epochs)
    return reader