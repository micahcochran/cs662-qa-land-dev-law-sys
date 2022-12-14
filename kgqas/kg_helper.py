"""
These are helper functions for template generation from the Knowledge Graph.
"""

# These are functions that need to be removed from the project. 
# The code is pretty horrible for these.

# Python Standard Libraries
import itertools
from pathlib import Path
import sys
from typing import Dict, Generator, Optional, List

# external libraries
import rdflib

# internal libraries
sys.path.append("..")  # hack to allow triplesdb imports
# from triplesdb.generate_template import generate_all_templates, generate_dimensional_templates, TemplateGeneration
import triplesdb.generate_template
# from triplesdb import TemplateGeneration

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

def generate_templates_shuffle(random_state: Optional[int] = None) \
        -> Generator[dict, None, None]:
    """build the templates"""
    kg = rdflib.Graph()
    kg.parse("../triplesdb/combined.ttl")

    tg_iter = triplesdb.generate_template.generate_all_templates_shuffle(kg, kg, random_state)

    return tg_iter

def generate_dim_templates(dimreq_kg: Optional[rdflib.graph.Graph] = None) -> itertools.chain[dict]:
    if dimreq_kg is None:
        dimreq_kg = rdflib.Graph()
        # one of the templates is unable to handle combined.ttl
        # dimreq_kg.parse("../triplesdb/bulk2.ttl")
        # kg = rdflib.Graph()
        dimreq_kg.parse("triplesdb/combined.ttl")
    template_path = Path('../triplesdb/templates')
    tg = triplesdb.generate_template.TemplateGeneration(template_path)

    # return triplesdb.generate_template.generate_dimensional_templates(dimreq_kg)
    return tg.generate_dimensional_templates(dimreq_kg)

def label_dictionary() -> dict:
    raise DeprecationWarning("Use template_number_dictionary() or template_names instead")
    tg_obj = triplesdb.generate_template.TemplateGeneration()
    return tg_obj.template_number_dict

#@property
def template_number_dictionary() -> Dict[str, int]:
    template_path = Path('../triplesdb/templates')
    tg_obj = triplesdb.generate_template.TemplateGeneration(template_path)
    return tg_obj.template_number_dict

# @property
def template_names() -> List[str]:
    tg_obj = triplesdb.generate_template.TemplateGeneration()
    return tg_obj.template_names()


def get_template(name: str) -> dict:
    tg_obj = triplesdb.generate_template.TemplateGeneration()
    return tg_obj.get_template(name)