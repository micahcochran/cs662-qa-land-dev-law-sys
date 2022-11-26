""" _summary_
"""
# %%
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from datasets import load_dataset
from datasets import load_metric
import evaluate
import numpy as np

file_path = f'{os.getcwd()}/data'

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

data_files = {"train": f'{file_path}/json/QAZoningTrain.json', "test": f'{file_path}/json/QAZoningTest.json'} # * this is how to load multiple files, need to sklearn train_test_split into two sets first
print(data_files)
QA_dataset = load_dataset('json', data_files=data_files)
print(QA_dataset)

tokenized_data = QA_dataset.map(preprocess_function, batched=True)
    
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=48)

from transformers import TrainingArguments, Trainer

metric = evaluate.load('f1')

training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # print(logits, labels)
    return metric.compute(predictions=predictions, references=labels, average='macro')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    compute_metrics=compute_metrics
)
# %%
trainer.train()
# %%
trainer.evaluate()
