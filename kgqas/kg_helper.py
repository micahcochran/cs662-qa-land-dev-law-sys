"""
These are helper functions for template generation from the Knowledge Graph.
"""

# Python Standard Libraries
import itertools
import sys

# external libraries
import rdflib

# internal libraries
sys.path.append("..")  # hack to allow triplesdb imports
from triplesdb.generate_template import generate_all_templates, TemplateGeneration


def generate_templates() -> itertools.chain[dict]:
    """build the templates"""
    uses_kg = rdflib.Graph()
    uses_kg.parse("../triplesdb/combined.ttl")

    dimreq_kg = rdflib.Graph()
    # one of the templates is unable to handle combined.ttl
    dimreq_kg.parse("../triplesdb/bulk2.ttl")

    tg_iter = generate_all_templates(uses_kg, dimreq_kg)

    return tg_iter
#        return generate_all_templates()

def label_dictionary() -> dict:
    tg_obj = TemplateGeneration()
    return tg_obj.template_number_dict