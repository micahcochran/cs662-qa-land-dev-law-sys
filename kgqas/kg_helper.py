"""
These are helper functions for template generation from the Knowledge Graph.
"""

# Python Standard Libraries
import itertools
import sys
from typing import Dict, Optional, List

# external libraries
import rdflib

# internal libraries
sys.path.append("..")  # hack to allow triplesdb imports
# from triplesdb.generate_template import generate_all_templates, generate_dimensional_templates, TemplateGeneration
import triplesdb.generate_template

# TODO A class singleton to store the KG would not be a bad idea.  It might reduce overhead.

def get_knowledge_graph() -> rdflib.Graph:
    """knowledge graph"""
    uses_kg = rdflib.Graph()
    uses_kg.parse("../triplesdb/combined.ttl")
    return uses_kg

def generate_templates() -> itertools.chain[dict]:
    """build the templates"""
    uses_kg = rdflib.Graph()
    uses_kg.parse("../triplesdb/combined.ttl")

    dimreq_kg = rdflib.Graph()
    # one of the templates is unable to handle combined.ttl
    dimreq_kg.parse("../triplesdb/bulk2.ttl")

    tg_iter = triplesdb.generate_template.generate_all_templates(uses_kg, dimreq_kg)

    return tg_iter
#        return generate_all_templates()

def generate_dim_templates(dimreq_kg: Optional[rdflib.graph.Graph] = None) -> itertools.chain[dict]:
    if dimreq_kg is None:
        dimreq_kg = rdflib.Graph()
        # one of the templates is unable to handle combined.ttl
        dimreq_kg.parse("../triplesdb/bulk2.ttl")

    return triplesdb.generate_template.generate_dimensional_templates(dimreq_kg)

# There are some distinctions in :ZoningDistrict and :ZoningDivisionDistrict that did not appear in the original
# generate_template.  This is causing a few questions that could be answerable, but it would have to use a :seeAlso
# hop.  This is doable, but out of scope for this phase of the project.  The function below is the workaround.
def remove_empty_answers(question_corpus: List[dict]) -> List[dict]:
    return [q for q in question_corpus if q['answer'] != []]

def label_dictionary() -> dict:
    raise DeprecationWarning("Use template_number_dictionary() or template_names instead")
    tg_obj = triplesdb.generate_template.TemplateGeneration()
    return tg_obj.template_number_dict

#@property
def template_number_dictionary() -> Dict[str, int]:
    tg_obj = triplesdb.generate_template.TemplateGeneration()
    return tg_obj.template_number_dict

# @property
def template_names() -> List[str]:
    tg_obj = triplesdb.generate_template.TemplateGeneration()
    return tg_obj.template_names()


def get_template(name: str) -> dict:
    tg_obj = triplesdb.generate_template.TemplateGeneration()
    return tg_obj.get_template(name)