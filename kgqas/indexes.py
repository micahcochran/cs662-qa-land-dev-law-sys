"""
Indexes are build from Knowledge Graph labels, permitted uses, and units.
"""
# internal Python libraries
import re
from typing import Set, Tuple

# external libraries
import rdflib

# TODO: Perhaps add numeric values specified in the KG to the index.

# For these indexes are in a set of tuples.
# The text label (Literal) is first element, and the URI fragment is second element.
# The URI fragment could be in any position, the Literal must be in the object position.
# for units it is unit's label, and units UCUM symbol


class IndexesKG:

    def __init__(self, kg=None):
        """kg - initialize with a knowledge graph"""
        if kg is None:
            kg = rdflib.Graph()
            # load the graph
            kg.parse("../triplesdb/combined.ttl")


        ### Another set of labels
        # For parsing these types of triples
        # :maxBuildingHeight rdfs:label "maximum building height"

        sparql_labels2 = """SELECT ?subject ?label WHERE {
            ?subject a ?type_of .
            ?subject rdfs:label ?label .
        }"""

        results_labels2 = kg.query(sparql_labels2)

        # results in a set
        self._label_index2 = set([(str(res.label), ':' + res.subject.fragment) for res in results_labels2])


        ### Permitted uses
        sparql_permitted_uses = """SELECT ?use WHERE {
            ?x :permitsUse ?use .
        }
        """
        results_uses = kg.query(sparql_permitted_uses)
#        self._permitted_uses_index = set([str(res.use) for res in results_uses])
        self._permitted_uses_index = set([(str(res.use), ':permitsUse') for res in results_uses])
#        self._permitted_uses_index = set([('', ':permitsUse', str(res.use)) for res in results_uses])

        self._units_index = self._init_units()

        self._numeric_index = self._init_numbers(kg)

        # combine the indexes
        self._all_indexes = self._label_index2 | self._permitted_uses_index | self._units_index | self._numeric_index


    def _init_units(self) -> Set[Tuple[str, str]]:

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
        return set(UNITS_NAME.items())

#    def _init_numbers(self, kg) -> Set[Tuple[str, str]]:
    def _init_numbers(self, kg) -> Set[Tuple[str, str]]:
        # The CDT typing causes problems for SPARQL to be able to do any filtering, so python has to do it instead.
        sparql_everything = """
SELECT ?pred ?obj WHERE {
    ?sub ?pred ?obj .
}
        """
        number_list = []
        sparql_results = kg.query(sparql_everything)
        for res in sparql_results:
            pattern = "^\d+"
            match = re.search(pattern, res.obj)
            if match:
#                print(f'{match}')
#                print(f'{res.obj[:match.end()]}')
#                print(':' + res.pred.fragment)
                # in the format of ('5', ':minSideSetback')
                number_list.append((res.obj[:match.end()], ':' + res.pred.fragment))
            # print(f'{r.pred} - {x.match}')
        # TODO These numbers may relate to multiple predicates, for the moment, I am going to ignore that
        #      and remove the predicates, so that I can only use the numbers.
        # print(number_list)
        return set([(num, '') for num, _ in number_list])

    @property
    def label_index2(self) -> set:
        return self._label_index2

    # These are a set of object which have the :permitsUse property.
    # We will always know that :permitsUse is a predicate, with the use as an object.
    @property
    def permitted_uses_index(self) -> set:
        return self._permitted_uses_index

    @property
    def numeric_index(self) -> set:
        return self._numeric_index

    @property
    def units_index(self) -> set:
        return self._units_index

    @property
    def all_indexes(self) -> set:
        return self._all_indexes

    def all_index_labels(self) -> list:
        return [label for label, _fragment in self._all_indexes]

# Intended only for testing
if __name__ == '__main__':
    indexkg = IndexesKG()

    print("==========  Label Index 2 ==========")
    print(indexkg.label_index2)
    print("==========  Numeric Index 2 ==========")
    print(indexkg.numeric_index)
    print("==========  Permitted Uses Index ==========")
    print(indexkg.permitted_uses_index)
    print("==========  Units Index ==========")
    print(indexkg.units_index)

    print(f'\nTotal number of items in the index: {len(indexkg.all_indexes)}')
