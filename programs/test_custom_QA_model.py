""" _summary_
"""

import os
import sys
import json
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append('../cs662-qa-land-dev-law-sys/')

from nlp.model import create_reader

data_name = 'example_squad'

cur_dir = os.getcwd()
model_dir = f'{cur_dir}/readers/{data_name}'

Path(model_dir).mkdir(parents=True, exist_ok=True)

reader = create_reader(model_dir, data_name, gpu=True, epochs=1)

json_data = json.loads(reader.data)

print(json_data['version'])