#!/usr/bin/env python3

"""
Code to generate templates
"""

# Python libraries
import itertools
from pathlib import Path
import random
import string
import sys
from typing import Dict, Generator, List, Optional, Union, Tuple
if sys.version_info < (3, 11):
    # tomli - external dependency for earlier python versions
    import tomli as tomllib
else:
    # internal library in Python 3.11+
    import tomllib


# Developed using rdflib version 6.2.0 is the current version as of 2022-10-31.
# using feature URIRef.fragment added in this version, but could easily program around if needed.
import rdflib


# FIXME Plural/singular form substitution needs implementation.
#           "Is a bank a permitted use?" versus "Are banks a permitted use?"


class QueryUsesSparql:
    """SPARQL queries that have multiple uses for the Permitted Uses Knowledge Graph."""

    def __init__(self, uses_kg: rdflib.graph.Graph):
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
    { ?zid    a           :ZoningDistrict .
      ?zid    rdfs:label  ?zoning_label .
    }
    UNION
    { ?zid    a           :ZoningDistrictDivision .
      ?zid    rdfs:label  ?zoning_label .
    }
}"""
        results = self.uses_kg.query(sparql)

        for zoning in set([str(res.zoning_label) for res in results]):
            yield zoning


    def all_uses_zoning_only_true_iter(self) -> Generator[Tuple[str, str], None, None]:
        """
        iterator of all the permitted uses in the knowledge graph.  These are all true result.

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


    # This causes Template 2 to have 3276 questions versus 403 of only the true questions
    # from .all_uses_zoning_only_true_iter().
    def all_uses_zoning_iter(self) -> Generator[Tuple[str, str], None, None]:
        """
        iterator of all the permitted uses in the knowledge graph.  These are true and false results.

        returns Generator that provides a Tuple with (use, zoning)
        """
        # Cartesian product of uses and zoning.
        # Sort to account for differences from how the data comes in from the knowledge graph.
        results = sorted(itertools.product(self.all_uses_iter(), self.all_zoning_iter()))

        for res in results:
#            print(f"RES: {res}")
            yield res


ZONING_RDF_PREFIX = 'http://www.example.org/ns/lu/zoning#'

# key is the unit's name, value is the unit per https://unitsofmeasure.org/ucum  in the designation c/s
UNITS_NAME = {
    # ---  area units  ---
    'acre': '[acr_us]',
    "square feet": "[sft_i]",

    # ---  Length units  ---
    "feet": '[ft_i]',

    # --- Custom units for Zoning ---
    'acres per dwelling unit': '[acr_us/du]',
    'dwelling units per acre': '[du/acr_us]',
    'units per acre': '[u/acr_us]',
}

# key is UCUM unit designation, value is the unit's name
UNITS_SYMBOL = {v: k for k, v in UNITS_NAME.items()}

# CDT datatypes for UCUM units
UNIT_DATATYPE = {
    # ---  area units  ---
    '[acr_us]': 'cdt:area',
    '[sft_i]': 'cdt:area',

    # ---  Length units  ---
    '[ft_i]': 'cdt:length',

    # ---  Custom units for Zoning  ---
    '[acr_us/du]': 'cdt:dimensionless',
    '[du/acr_us]': 'cdt:dimensionless',
    '[u/acr_us]': 'cdt:dimensionless',
}


class QueryDimensionsSparql:
    """SPARQL queries that have multiple uses for the Dimensional Requirements Knowledge Graph."""

    def __init__(self, dimensional_kg: rdflib.graph.Graph):
        self.dimensional_kg = dimensional_kg

        # dimensional regulations
        # key is text, value is the name of predicate in the dimensional requirements KG
        self.DIM_REGULATIONS_TEXT = self._get_kg_properties()

        # the key is the predicate, the values is the text description
        self.DIM_REGULATIONS_PRED = {v: k for k, v in self.DIM_REGULATIONS_TEXT.items()}

        # the key is the predicate with full URI, the values is the text description
        self.DIM_REGULATIONS_PRED_URI = {ZONING_RDF_PREFIX + v[1:]: k for k, v in self.DIM_REGULATIONS_TEXT.items()}

        # DIM_REGULATIONS_TEXT_URI = {v:k for k,v in DIM_REGULATIONS_TEXT.items()}

    def all_zoning_iter(self) -> Generator[str, None, None]:
        """
        iterator of all the zoning districts in the knowledge graph
        """
        sparql = """
SELECT ?zoning_label
WHERE {
    { ?zid    a           :ZoningDistrict .
      ?zid    rdfs:label  ?zoning_label . 
    } 
    UNION
    { ?zid    a           :ZoningDistrictDivision .
      ?zid    rdfs:label  ?zoning_label . 
    } 
}"""

        results = self.dimensional_kg.query(sparql)

        for zoning in set([str(res.zoning_label) for res in results]):
            yield zoning

    def all_zoning_dims_iter(self):
        """
        iterator of all the zoning division districts in the knowledge graph
        These are the values that should have dimensions.
        """

        sparql = """
SELECT ?zoning_label
WHERE {
    # Get the :ZoningDistrict
    {
        ?zid    a           :ZoningDistrict .
        ?zid    rdfs:label  ?zoning_label .
        # remove :ZoningDistrict that are the subject of a :seeAlso tag
        FILTER NOT EXISTS {
            ?ozid    rdfs:seeAlso    ?zid .
        }
    }
    UNION
    # Get the :ZoiningDistrictDivision values
    {
        ?zid    a           :ZoningDistrictDivision .
        ?zid    rdfs:label  ?zoning_label .
    }
}
"""
        results = self.dimensional_kg.query(sparql)

        for zoning in set([str(res.zoning_label) for res in results]):
            yield zoning


    def all_regulations_zoning_dims_iter(self) -> Generator[Tuple[str, str, str], None, None]:
        """
        iterator
        :return:
        """
        for zoning_div in self.all_zoning_dims_iter():
            for regulation_text, regulation_predicate in self.DIM_REGULATIONS_TEXT.items():
                yield regulation_predicate, regulation_text, zoning_div

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
            if str(regulation_predicate) not in self.DIM_REGULATIONS_PRED_URI:
                continue

            # Skip regulation_values that where the 1st token does not start with numbers.
            # Note: SPARQL provides an isNumeric, but that does not work with values like "10 [ft_i]"^^cdt:length.
            if not regulation_value.split()[0].isnumeric():
                continue

            # this is regulation_predicate, regulation_text, regulation_value, zoning_label
            yield ':' + regulation_predicate.fragment, self.DIM_REGULATIONS_PRED_URI[
                str(regulation_predicate)], regulation_value, zoning_label


    def _get_kg_properties(self) -> dict:
        """generates Knowledge Graph's properties
            specifically for DIM_REGULATIONS_TEXT variable from the graph
           returns dictionary"""
        sparql_properties = """
SELECT ?property_name ?property_label

WHERE {
    ?property_name a rdf:Property ;
                rdfs:label  ?property_label .
}
        """
        results = self.dimensional_kg.query(sparql_properties)
        # This returns the fragment form (example :minLotSize)
        # and does no checking if the property is from the correct KG.
        kg_properties_set = {str(row.property_label): ':' + row.property_name.fragment for row in results}
        #       print(kg_properties_set)
        return kg_properties_set


class TemplateGeneration:
    """
    Generates templates questions with SPQARL queries from the knowledge graph.
    """

    def __init__(self, template_path: Path = Path('templates')):
        # print('TemplateGeneration.__init__()')
        self.templates = {}

        # print(f'Path().cwd(): {Path().cwd()}')
        # print(f'template_path: {template_path}')

        # load TOML templates from the template folder
        template_path_objs = list(template_path.glob('template*.toml'))

        # print(f'template_path_objs: {template_path_objs}')
        # print(f'len(template_path_objs): {len(template_path_objs)}')

        if len(template_path_objs) == 0:
            raise RuntimeError('Code is having problems locating the templates.')
#            print("Error")

        for t in template_path_objs:
            name, template_dict = self._load_template(t)
            self.templates[name] = template_dict

        # must start at zero for xgboost labels.
        self._template_number = {tmplt: i for i, tmplt in enumerate(sorted(self.templates.keys()))}

    def _load_template_filename(self, filename: str) -> Tuple[str, dict]:
        template = None
        with open(filename, mode='rb') as fp:
            template = tomllib.load(fp)

        template_name = template['template']['template_name']

        return template_name, template['template']

    def _load_template(self, path: Path) -> Tuple[str, dict]:
        template = None
        with path.open(mode='rb') as fp:
            template = tomllib.load(fp)

        template_name = template['template']['template_name']

        return template_name, template['template']

    def template_names(self) -> List[str]:
        """returns the template names as a list
        this is in the same order as template_number_dict"""
        return list(sorted(self.templates.keys()))

    @property
    def template_number_dict(self) -> Dict[str, int]:
        """This has {template_name: number}.  Numbering starts at zero"""
        return self._template_number



    # NOTE: The dictionary was a design decision to allow extension
    # to add other variables.
    def generate_output(self, kg: rdflib.graph.Graph, template_name) -> Generator[dict, None, None]:
        """
        generate the templates for the permitted uses
        kg - knowledge graph,
        template_name - name of the template to use.

        return a generator that returns a dictionary.
        """
        # print(f'generate_output() with template_name {template_name}')
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
#                ('regulation_predicate', 'regulation_text', 'zoning'): qdims.all_regulations_zoning_iter(),
                ('regulation_predicate', 'regulation_text', 'zoning_dims'): qdims.all_regulations_zoning_dims_iter(),
                ('zoning',): qdims.all_zoning_iter(),
            }
        else:
            raise RuntimeError(
                f'self.templates["{template_name}"], knowledge_graph has an unknown value: {self.templates[template_name]["knowledge_graph"]}')

        # set up the templates
        sparql_template = string.Template(self.templates[template_name]['sparql_template'])
        question_templates = [string.Template(q_tmpl) for q_tmpl in self.templates[template_name]['question_templates']]

        variable_names = tuple(self.templates[template_name]['variables'])
        # print(f'variable_names: {variable_names}')
        for variables in iterators[variable_names]:
            # print(f"variable_names: {variable_names}, variables: {variables}")
            # make sure the variables that come in are in a tuple
            if isinstance(variables, tuple):
                variables_tuple = variables
            else:
                variables_tuple = (variables,)

            # Creates a dictionary that puts the variable names together with their values
            #   Make sure that all values in variable_tuple have been converted to strings.
            varibs = dict(zip(variable_names,
                              map(lambda x: str(x), variables_tuple)))

            # handle units for variable_values
            if 'regulation_value' in varibs \
                    and '[' in varibs['regulation_value'] and ']' in varibs['regulation_value']:
                value, unit_symbol = varibs['regulation_value'].split()
                # Example: "1000 [ft_i]"
                varibs['regulation_value'] = value  # = "1000"
                # print(f"value is {value}")
                varibs['unit_symbol'] = unit_symbol  # = "[ft_i]"
                varibs['unit_text'] = UNITS_SYMBOL[unit_symbol]  # = "feet"
                varibs['unit_datatype'] = UNIT_DATATYPE[unit_symbol]  # = "cdt:length"

            # print(f"varibs: {varibs}")

            result = {'sparql': sparql_template.substitute(varibs),
                      'template_name': template_name,
                      'variables': varibs,
                      }

            # print(result)
            # execute SPARQL and create answer.
            result['answer'] = self.execute_sparql_for_answer(kg, result['sparql'],
                                                              self.templates[template_name]['variable_names_sparql'],
                                                              self.templates[template_name]['answer_datatype'])

            for q_template in question_templates:
                # print(f"varibs = {varibs}")
                # print(f"templates are {self.templates[template_name]['question_templates']}")
                result['question'] = q_template.substitute(varibs)
                yield result

    def get_template(self, name: str) -> dict:
        """provide the dictionary of the template"""
        return self.templates[name]

    def execute_sparql_for_answer(self, kg: rdflib.graph.Graph, sparql: str, varibs: tuple,
                                  expected_result) -> Union[bool, list]:
        """execute SPARQL query get the answer, returns either a list or boolean"""
        # print(f'sparql: {sparql}')
        # print(f'varibs: {varibs}')
        # print(f'expected_result: {expected_result}')
        result = kg.query(sparql)

        # Assumption this is a one or zero variable answer from SPARQL
        if len(varibs) > 1:
            raise ValueError(f'varibs should have only one value in the tuple: varibs = {varibs} ')
        elif expected_result == 'list':
#        elif expected_result == list:
            return [str(r[varibs[0]]) for r in result]
#        elif expected_result == bool:
        elif expected_result == 'bool':
            return result.askAnswer
        else:
            raise ValueError(
                f"expected_result parameter should be either a 'list' or 'boolean', but is listed as '{expected_result}'")



    def generate_all_templates(self, uses_kg: Optional[rdflib.graph.Graph] = None,
                            dimreq_kg: Optional[rdflib.graph.Graph] = None) -> itertools.chain[dict]:
        """generate all the templates

        uses_kg or dimreq_kg rdflib.Graph() objects may be passed.  Otherwise, these will be loaded automatically."""
        print('generate_all_templates()')
        if uses_kg is None:
            uses_kg = rdflib.Graph()
            # load the graph related to the permitted uses
            # uses_kg.parse("permits_use2.ttl")
            uses_kg.parse("combined.ttl")

        if dimreq_kg is None:
            dimreq_kg = rdflib.Graph()
            # load the graph related to the permitted uses
            #    dimreq_kg.parse("bulk.ttl")
            dimreq_kg.parse("bulk2.ttl")
            # dimreq_kg.parse("combined.ttl")
        # print(f'len(dimreq_kg): {len(dimreq_kg)}, len(uses_kg): {len(uses_kg)}')

        # tg = TemplateGeneration()
        iterators = []
        for template_name in self.template_names():
            kg_to_use = self.get_template(template_name)['knowledge_graph']
            if kg_to_use == 'dimensional_reqs':
                iterators.append(self.generate_output(dimreq_kg, template_name))
            elif kg_to_use == 'permitted_uses':
                iterators.append(self.generate_output(uses_kg, template_name))
            else:
                raise RuntimeError(
                    f'self.templates["{template_name}"], knowledge_graph has an unknown value: {kg_to_use}')

        return itertools.chain.from_iterable(iterators)


    def generate_all_templates_shuffle(self, uses_kg: Optional[rdflib.graph.Graph] = None,
                            dimreq_kg: Optional[rdflib.graph.Graph] = None,
                            random_state: Optional[int] = None) -> Generator[dict, None, None]:
        """shuffled version of generate_all_templates().

        Note: this will use more memory than generate_all_templates. It has to store the shuffled list."""
        random.seed(random_state)
        results = list(self.generate_all_templates(uses_kg=uses_kg, dimreq_kg=dimreq_kg))
        # Apparently, dictionaries cannot be sorted.
    #    results = sorted(generate_all_templates(uses_kg=uses_kg, dimreq_kg=dimreq_kg))
        # print(f'RESULTS: {results}')
        random.shuffle(results)
        # print(f'RESULTS: {results}')

        for res in results:
        #    print(f'RES: {res}')
            yield res


    def generate_dimensional_templates(self, dimreq_kg: Optional[rdflib.graph.Graph] = None) -> itertools.chain[dict]:
        if dimreq_kg is None:
            dimreq_kg = rdflib.Graph()
            # load the graph related to the permitted uses
            #    dimreq_kg.parse("bulk.ttl")
            dimreq_kg.parse("bulk2.ttl")
            # dimreq_kg.parse("combined.ttl")

        # tg = TemplateGeneration()
        iterators = []
        for template_name in self.template_names():
            kg_to_use = self.get_template(template_name)['knowledge_graph']
            if kg_to_use == 'dimensional_reqs':
                iterators.append(self.generate_output(dimreq_kg, template_name))

        return itertools.chain.from_iterable(iterators)


def main() -> int:
    import argparse

    selection_epilog = """
  selection options:
     1 - Template 1 - template_use_1var_m_answer
     2 - Template 2 - template_use_2var_yn_answer
     3 - Template 3 - template_dimreg_2var_m_answer
     4 - Template 4 - template_dimreg_4var_yn_answer
     5 - Template 5 - template_use_1var_yn_answer
     all - output all the templates"""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=selection_epilog)

    parser.add_argument('selection')
    parser.add_argument('-q', '--question-only', action='store_true', help='show only the question')
    parser.add_argument('-r', '--randomize', action='store_true', help='randomize entries')
    args = parser.parse_args()

    kg = rdflib.Graph()
    kg.parse("combined.ttl")

    tg = TemplateGeneration()

    # TODO needs to print this out as part of the message.
    templates = ['template_use_1var_m_answer',
                 'template_use_2var_yn_answer',
                 'template_dimreg_2var_m_answer',
                 'template_dimreg_4var_yn_answer',
                 'template_use_1var_yn_answer',]

    # print(f"ARGS: {args}")

    if args.selection.lower() == 'all':
        if args.randomize:
            template_iter = tg.generate_all_templates_shuffle(kg, kg)
        else:
            template_iter = tg.generate_all_templates(kg, kg)
    elif args.selection.isnumeric():
        if args.randomize:
            print("Randomize flag only works with selection 'all'.")
            return 3

        idx = int(args.selection)-1
        if idx > len(templates):
            print(f'Template selection value of "{args.selection}" is larger than the array\'s length {len(templates)}.')
            return 1

        template_iter = tg.generate_output(kg, templates[idx])
    else:
        print(f'"{args.selection}" is not a valid value for selection.')
        return 2

    if args.question_only:
        for d in template_iter:
            print(d['question'])
    else:
        # print dictionary
        for d in template_iter:
            print(d)

    return 0


if __name__ == '__main__':
    sys.exit(main())
