#!/usr/bin/env python3

"""
Code to generate templates
"""

from typing import Generator, Iterator, Tuple
import sys
from string import Template

# Developed using rdflib version 6.2.0 is the current version as of 2022-10-31.
# using feature URIRef.fragment added in this version, but could easily program around if needed.
import rdflib


# SPARQL queries that are useful
class QueryUsesSparql:
    """SPARQL queries that have multiple uses for the Permitted Uses Knowledge Graph."""

    def __init__(self, uses_kg):
        self.uses_kg = uses_kg

    def all_uses_iter(self) -> Generator[str, None, None]:
        """
        iterator of all the permitted uses in the knowledge graph
        """
        sparql = """
        SELECT ?use

        WHERE {
                ?zoning :permitsUse ?use .
        }
        """

        results = self.uses_kg.query(sparql)

        for use in set([str(res.use) for res in results]):
            yield use

    def all_zoning_iter(self):
        """
        iterator of all the zoning districts in the knowledge graph
        """

        sparql = """
        SELECT ?zoning_label

        WHERE {
                ?zoning rdfs:label ?zoning_label .
        }
        """

        results = self.uses_kg.query(sparql)

        for zoning in set([str(res.zoning_label) for res in results]):
            yield zoning

    def all_uses_zoning_iter(self) -> Generator[Tuple[str, str], None, None]:
        """
        iterator of all the permitted uses in the knowledge graph

        returns tuple with (use, zoning)
        """
        sparql = """
        SELECT ?zoning_label ?use

        WHERE {
                ?zoning :permitsUse ?use .
                ?zoning rdfs:label ?zoning_label .
        }
        """

        results = self.uses_kg.query(sparql)

        for res in results:
            yield res.use, res.zoning_label


ZONING_RDF_PREFIX = 'http://www.example.org/ns/lu/zoning#'

# dimensional regulations
# key is text, value is the name of predicate in the dimensional requirements KG
DIM_REGULATIONS_TEXT = {"minimum lot size": ":minLotSize",
                        "maximum density": ":maxDensity",   # FIXME ignore maxDensity temporarily, so it does not block progress
                        "minimum lot width": ":minLotWidth",
                        "minimum lot depth": ":minLotDepth",
                        "minimum front setback": ":minFrontSetback",
                        "minimum side setback": ":minSideSetback",
                        "minimum rear setback": ":minRearSetback",
                        "maximum building height": ":maxBuildingHeight", }

# the key is the predicate, the values is the text description
DIM_REGULATIONS_PRED = {v: k for k, v in DIM_REGULATIONS_TEXT.items()}

# the key is the predicate with full URI, the values is the text description
DIM_REGULATIONS_PRED_URI = {ZONING_RDF_PREFIX + v[1:]: k for k, v in DIM_REGULATIONS_TEXT.items()}

# DIM_REGULATIONS_TEXT_URI = {v:k for k,v in DIM_REGULATIONS_TEXT.items()}

# key is the unit's name, value is the unit per https://unitsofmeasure.org/ucum  in the designation c/s
UNITS_NAME = {
    # ---  area units  ---
    'acre': '[acr_us]',
    "square feet": "[sft_i]",

    # ---  Length units  ---
    "feet": '[ft_i]',

    # --- Custom units for Zoning ---
    'acres per dwelling unit': '[acr_u/du]',
    'dwelling units per acre': '[du/acr_u]',
    'units per acre': '[u/acr_u]',
}

# key is UCUM unit designation, value is the unit's name
UNITS_SYMBOL = {v: k for k, v in UNITS_NAME.items()}


class QueryDimensionsSparql:
    """SPARQL queries that have multiple uses for the Dimensional Requirements Knowledge Graph."""

    def __init__(self, dimensional_kg):
        self.dimensional_kg = dimensional_kg

    def all_zoning_iter(self) -> Generator[str, None, None]:
        """
        iterator of all the zoning districts in the knowledge graph
        """

        sparql = """
        SELECT ?zoning_label

        WHERE {
                ?zoning rdfs:label ?zoning_label .
        }
        """

        results = self.dimensional_kg.query(sparql)

        for zoning in set([str(res.zoning_label) for res in results]):
            yield zoning

    def all_regulations_zoning_iter(self) -> Generator[Tuple[str, str, str], None, None]:
        """
        iterator
        :return:
        """
        for zoning in self.all_zoning_iter():
            for regulation_text, regulation_predicate in DIM_REGULATIONS_TEXT.items():
                yield regulation_predicate, regulation_text, zoning

    def all_regulations_values_zoning_iter(self):
        """
        iterator of regulation_predicate, regulation_value, zoning districts in the knowledge graph

        units are left in regulation_value
        """

        sparql = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://www.example.org/ns/lu/zoning#>

        SELECT ?regulation_predicate ?regulation_value ?zoning_label

        WHERE {
                ?zoning rdfs:label ?zoning_label ;
                        ?regulation_predicate ?regulation_value .
        }
        """

        results = self.dimensional_kg.query(sparql)

        for regulation_predicate, regulation_value, zoning_label in results:

            # skip predicates that are not in the dictionary of dimensional regulations
            if str(regulation_predicate) not in DIM_REGULATIONS_PRED_URI:
                continue
            # REMOVEME
            # print(type(regulation_value))
            # print(dir(regulation_value))
            # print(regulation_value._datatype)

            #            yield DIM_REGULATIONS_PRED[':' + regulation_predicate.fragment], regulation_predicate, regulation_value, zoning_label
            # this is regulation_predicate, regulation_text, regulation_value, zoning_label
            yield ':' + regulation_predicate.fragment, DIM_REGULATIONS_PRED_URI[
                str(regulation_predicate)], regulation_value, zoning_label


class TemplateGeneration:
    """
    Generates templates questions with SPQARL queries from the knowledge graph.
    """

    def __init__(self):
        self.templates = {}
        # ===  PERMITTED USE TEMPLATES  ====================================
        # Template 1 - template_use_1var_m_answer
        t1 = {'template_name': 'template_use_1var_m_answer',
              'knowledge_graph': 'permitted_uses',
              'variables': ('use',),  # variables feed into the templates
              'variable_names_sparql': ('zoning',),  # variable resulting from SPARQL query
              'sparql_template': """
SELECT ?zoning_label

WHERE {
        ?zoning :permitsUse "${use}" .
        ?zoning rdfs:label ?zoning_label .
}
""",
              'question_templates': ["Which zoning districts allow ${use}?",
                                     "Which zoning districts permit ${use}?",
                                     "I would like to build ${use}.  Which zoning districts permits this use?"],
              'answer_datatype': list,
              #              'iter_method' :
              }
        self.templates['template_use_1var_m_answer'] = t1

        # Template 2 - template_use_2var_yn_answer
        t2 = {'template_name': 'template_use_2var_yn_answer',
              'knowledge_graph': 'permitted_uses',
              'variables': ('use', 'zoning'),  # variables feed into the templates
              'variable_names_sparql': tuple(),  # variable resulting from SPARQL query
              'sparql_template': """
ASK {
    ?zoning :permitsUse "${use}" ;
            rdfs:label "${zoning}" .
}
""",
              'question_templates': ["Are $use allowed in a $zoning zoning district?",
                                     "Are $use permitted in a $zoning zoning district?"],
              'answer_datatype': bool,
              }

        self.templates['template_use_2var_yn_answer'] = t2

        # Template 5 - template_use_1var_yn_answer
        t5 = {'template_name': 'template_use_1var_yn_answer',
              'knowledge_graph': 'permitted_uses',
              'variables': ('use',),  # variables feed into the templates
              'variable_names_sparql': tuple(),  # variable resulting from SPARQL query
              'sparql_template': """
ASK {
        ?zoning :permitsUse "${use}" .
}
""",
              'question_templates': ["Are $use permitted?",
                                     "Are $use allowed?"],
              'answer_datatype': bool,
              }

        self.templates['template_use_1var_yn_answer'] = t5
        # ===  DIMENSIONAL REQUIREMENT TEMPLATES  ==================================
        # Template 3 - template_dimreg_2var_m_answer
        t3 = {'template_name': 'template_dimreg_2var_m_answer',
              'knowledge_graph': 'dimensional_reqs',
              'variables': ('regulation_predicate', 'regulation_text', 'zoning'),  # variables feed into the templates
              'variable_names_sparql': ('regulation_value',),  # variable resulting from SPARQL query
              'sparql_template': """
SELECT ?regulation_value

WHERE {
        ?zoning $regulation_predicate ?regulation_value ;
                rdfs:label "${zoning}" .
}
    """,
              'question_templates': ["What is the $regulation_text in the $zoning zoning district?"],
              'answer_datatype': list,
              }

        self.templates['template_dimreg_2var_m_answer'] = t3

        # Template 4 - template_dimreg_4var_yn_answer
        #        The units are a pretty small group:
        #           From the regulations: feet, square feet, acres
        #           Other units that are similar measures: inches, meters, centimeters, square meters, hectares
        #           All of these units have many ways to abbreviate.
        #
        #  SPARQL queries don't care about the units, I've tried this out in rdflib and Jena.
        #  Some code will be needed to deals with units in order to make sure to respond with the right units if conversion
        #  is needed.
        # TODO may need to somehow make $unit_text work with the rest of the code.
        t4 = {'template_name': 'template_dimreg_4var_yn_answer',
              'knowledge_graph': 'dimensional_reqs',
              'variables': ('regulation_predicate', 'regulation_text', 'regulation_value', 'zoning'),
              # variables feed into the templates
              'variable_names_sparql': tuple(),  # variable resulting from SPARQL query
              'sparql_template': """
ASK {
        ?zoning $regulation_predicate ${regulation_value};
                rdfs:label "${zoning}" .
}
    """,
              #          'question_templates':
              #                ["Is the $regulation_text for a property in the $zoning zoning district $regulation_value square feet?"],
              'question_templates':
                  [
                      "Is the $regulation_text for a property in the $zoning zoning district $regulation_value $unit_text?"],

              'answer_datatype': bool,
              }

        self.templates['template_dimreg_4var_yn_answer'] = t4

    def template_names(self) -> list:
        """template names"""
        return self.templates.keys()

    # NOTE: The dictionary was a design decision to allow extension
    # to add other variables.
    def generate_output(self, kg, template_name) -> Generator[dict, None, None]:
        """
        generate the templates for the permitted uses
        kg - knowledge graph,
        template_name - name of the template to use.

        return a generator that returns a dictionary.
        """
        if template_name not in self.templates:
            raise ValueError(f'template name: {template_name} does not exist.')

        # a passive way to error check
        assert (self.templates[template_name]['knowledge_graph'] in ('dimensional_reqs', 'permitted_uses'))

        # this is just used to get information about variables,
        # not to make queries of the KG.
        if self.templates[template_name]['knowledge_graph'] == 'permitted_uses':
            qus = QueryUsesSparql(kg)

            iterators = {
                ('use',): qus.all_uses_iter(),
                ('use', 'zoning'): qus.all_uses_zoning_iter(),
                ('zoning',): qus.all_zoning_iter(),
            }
        elif self.templates[template_name]['knowledge_graph'] == 'dimensional_reqs':
            qdims = QueryDimensionsSparql(kg)

            # NOTE: variables should be in the same order as the tuples that come out of the iterator.
            iterators = {
                ('regulation_predicate', 'regulation_text', 'regulation_value', 'zoning'):
                    qdims.all_regulations_values_zoning_iter(),
                ('regulation_predicate', 'regulation_text', 'zoning'): qdims.all_regulations_zoning_iter(),
                ('zoning',): qdims.all_zoning_iter(),
            }
        else:
            raise RuntimeError(
                f'self.templates["{template_name}"], knowledge_graph has an unknown value: {self.templates[template_name]["knowledge_graph"]}')

        # set up the templates
        sparl_template = Template(self.templates[template_name]['sparql_template'])
        question_templates = [Template(q_tmpl) for q_tmpl in self.templates[template_name]['question_templates']]

        variable_names = self.templates[template_name]['variables']
        for variables in iterators[variable_names]:
            #   print(f"variable_names: {variable_names}, variables: {variables}")
            # make suer the variables  that come in are in a tuple
            if isinstance(variables, tuple):
                variables_tuple = variables
            else:
                variables_tuple = (variables,)

            # this creates a dictionary that puts the variable names together with their values
            varibs = dict(zip(variable_names, variables_tuple))
            # print(f"varibs: {varibs}")

            # print(f"'regulation_value' in varibs: {'regulation_value' in varibs}")
#            print(f"'[' in varibs.get('regulation_value'): {'[' in varibs.get('regulation_value')}")
#            print(f"']' in varibs.get('regulation_value'): {']' in varibs.get('regulation_value')}")

            # handle units for variable_values
            if 'regulation_value' in varibs \
                    and '[' in varibs['regulation_value'] and ']' in varibs['regulation_value']:
                value, unit_symbol = varibs['regulation_value'].split()
                # Example: "1000 [ft_i]"
                varibs['regulation_value'] = value  # = "1000"
                # print(f"value is {value}")
                varibs['unit_symbol'] = unit_symbol  # = "[ft_i]"
                varibs['unit_text'] = UNITS_SYMBOL[unit_symbol]  # = "feet"
                # TODO next step for handling units

            result = {'sparql': sparl_template.substitute(varibs),
                      # 'template_name': template_name,  # This could be added here
                      'varibles': varibs,
                     }

            for q_template in question_templates:
                # print(f"varibs = {varibs}")
                # print(f"templates are {self.templates[template_name]['question_templates']}")
                result['question'] = q_template.substitute(varibs)
                yield result


def main() -> int:
    uses_kg = rdflib.Graph()
    # load the graph related to the permitted uses
    uses_kg.parse("permits_use2.ttl")

    # import rdflib
    dimreq_kg = rdflib.Graph()
    # load the graph related to the permitted uses
    #    dimreq_kg.parse("bulk.ttl")
    dimreq_kg.parse("bulk2.ttl")

    tg = TemplateGeneration()
    if len(sys.argv) < 2:
        print_help()
        return 0
    elif sys.argv[1] == '1':
        print('=== template_use_1var_m_answer ===')
        template_iter = tg.generate_output(uses_kg, 'template_use_1var_m_answer')
    elif sys.argv[1] == '2':
        print('=== template_use_2var_yn_answer ===')
        template_iter = tg.generate_output(uses_kg, 'template_use_2var_yn_answer')
    elif sys.argv[1] == '3':
        print('=== template_dimreg_2var_m_answer ===')
        template_iter = tg.generate_output(dimreq_kg, 'template_dimreg_2var_m_answer')
    elif sys.argv[1] == '4':
        print('=== template_dimreg_4var_yn_answer ===')
        template_iter = tg.generate_output(dimreq_kg, 'template_dimreg_4var_yn_answer')
    elif sys.argv[1] == '5':
        print('=== template_use_1var_yn_answer ===')
        template_iter = tg.generate_output(uses_kg, 'template_use_1var_yn_answer')
    else:
        print_help()
        return 0

    # Currently, just printing a dictionary
    for d in template_iter:
        print(d)

    return 0


def print_help():
    print("""
    run with:
        ./generate_template.py [template_num]
                1 - Template 1 - template_use_1var_m_answer
                2 - Template 2 - template_use_2var_yn_answer
                3 - Template 3 - template_dimreg_2var_m_answer
                4 - Template 4 - template_dimreg_4var_yn_answer
                5 - Template 5 - template_use_1var_yn_answer
    """)


if __name__ == '__main__':
    sys.exit(main())
