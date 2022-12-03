# Zoning KGQAS

This is a Knowledge Graph Question Answering System it is somewhat based on the paper:

S. Aghaei, E. Raad and A. Fensel, "Question Answering Over Knowledge Graphs: A Case Study in Tourism," in IEEE Access, vol. 10, pp. 69788-69801, 2022, doi: 10.1109/ACCESS.2022.31871

Instead of tourism this is using the method for a zoning information Question Answering System (QAS).

### Files
* [KGQAS.ipynb](KGQAS.ipynb) - Start here.  This is a very high level view of the Zoning KGQAS working.
* [semantic_parsing.py](semantic_parsing.py) - This ties together the entire Semantic parsing phase from the other classses the below [Semantic Parsing](#Semantic_Parsing) section.
* [indexes.py](indexes.py) - The indexes are used by the Semantic Parsing class for training and classification.
* [kg_helper.py](kg_helper.py) - This is helper code for this code accessing the template generation.

## Semantic Parsing
These are listed in the order of their location in the Semantic Parsing phase.

1. Question Classification - [question_classification.py](question_classification.py)
2. Entity Linking and Class Linking - [entity_class_linking.py](entity_class_linking.py)
3. Relation Extraction - [relation_extraction.py](relation_extraction.py)
4. Slot Filling and Query Execution - [semantic_parsing.py](semantic_parsing.py)

## Knowledge Graph
This is located in the folder [../triplesdb](../triplesdb).
