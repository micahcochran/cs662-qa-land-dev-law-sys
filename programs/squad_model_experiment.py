""" _summary_
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from haystack.document_stores import InMemoryDocumentStore

document_store = InMemoryDocumentStore()

from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http

# Let's first get some documents that we want to query
# Here: 517 Wikipedia articles for Game of Thrones
doc_dir = f"{os.getcwd()}/programs/data/text"

print(doc_dir)

# convert files to dicts containing documents that can be indexed to our datastore
# You can optionally supply a cleaning function that is applied to each doc (e.g. to remove footers)
# It must take a str as input, and return a str.
docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

# We now have a list of dictionaries that we can write to our document store.
# If your texts come from a different source (e.g. a DB), you can of course skip convert_files_to_dicts() and create the dictionaries yourself.
# The default format here is: {"name": "<some-document-name>", "content": "<the-actual-text>"}

# Let's have a look at the first 3 entries:
# print(docs[:3])

# Now, let's write the docs to our DB.
document_store.write_documents(docs)

from haystack.nodes import TfidfRetriever

retriever = TfidfRetriever(document_store=document_store)

from haystack.nodes import FARMReader

# Load a  local model or any of the QA models on
# Hugging Face's model hub (https://huggingface.co/models)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

from haystack.pipelines import ExtractiveQAPipeline

pipe = ExtractiveQAPipeline(reader, retriever)

prediction = pipe.run(
    query="What fence materials are allowed in a R-1 district", params={"Retriever": {"top_k": 1}, "Reader": {"top_k": 1}}
)

from pprint import pprint

pprint(prediction['answers'])