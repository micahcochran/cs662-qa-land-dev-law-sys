"""
Indexes are build from rdf labels and permitted uses.
"""

# external libraries
import rdflib

# For these indexes are in a set of tuples.
# The text label (Literal) is first element, and the URI fragment is second element.
# The URI fragment could be in any position, the Literal must be in the object position.

class IndexesKG:

    def __init__(self):
        kg = rdflib.Graph()
        # load the graph
        kg.parse("../triplesdb/combined.ttl")

        # NOTE: This first index doesn't seem to be useful.
        # intended to parse these triples
        #  :r1 a           :ZoningDistrict .
        #  :r1 rdfs:label  "R1" .
#        sparql_labels = """SELECT ?label ?type_of WHERE {
#            ?subject a ?type_of .
#            ?subject rdfs:label ?label .
#        }"""

#        results_labels = kg.query(sparql_labels)

        # this results in a set
#       self._label_index = set([(str(res.label), ':' + res.type_of.fragment) for res in results_labels])

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

        # combine the indexes
        self._both_index = self._label_index2 | self._permitted_uses_index
    # currently this is a set, not sure if this should have another type
    # It is a set of tuples.  The first element of tuple is string for the label,
    # and the second element of the tuple is the Property.
#    @property
#    def label_index(self) -> set:
#        return self._label_index

    @property
    def label_index2(self) -> set:
        return self._label_index2

    # These are a set of object which have the :permitsUse property.
    # We will always know that :permitsUse is a predicate, with the use as an object.
    @property
    def permitted_uses_index(self) -> set:
        return self._permitted_uses_index

    @property
    def both_index(self) -> set:
        return self._both_index

    def both_index_labels(self) -> list:
        return [label for label, _fragment in self._both_index]

# Intended only for testing
if __name__ == '__main__':
    indexkg = IndexesKG()
#    print("==========  Label Index  ==========")
#    print(indexkg.label_index)
    print("==========  Label Index 2 ==========")
    print(indexkg.label_index2)
    print("==========  Permitted Uses Index ==========")
    print(indexkg.permitted_uses_index)
