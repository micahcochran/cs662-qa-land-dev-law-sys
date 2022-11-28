"""
Indexes are build from Knowledge Graph labels, permitted uses, and units.
"""
# internal Python libraries
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

        # combine the indexes
        self._all_indexes = self._label_index2 | self._permitted_uses_index | self._units_index
    # currently this is a set, not sure if this should have another type
    # It is a set of tuples.  The first element of tuple is string for the label,
    # and the second element of the tuple is the Property.
#    @property
#    def label_index(self) -> set:
#        return self._label_index

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

    @property
    def label_index2(self) -> set:
        return self._label_index2

    # These are a set of object which have the :permitsUse property.
    # We will always know that :permitsUse is a predicate, with the use as an object.
    @property
    def permitted_uses_index(self) -> set:
        return self._permitted_uses_index

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
#    print("==========  Label Index  ==========")
#    print(indexkg.label_index)
    print("==========  Label Index 2 ==========")
    print(indexkg.label_index2)
    print("==========  Permitted Uses Index ==========")
    print(indexkg.permitted_uses_index)
    print("==========  Units Index ==========")
    print(indexkg.units_index)

    print(f'\nTotal number of items: {len(indexkg.all_indexes)}')
