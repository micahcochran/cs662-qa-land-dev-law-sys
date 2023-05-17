# Description of files

These are files that make up a triples "Database" in Turtle RDF format (.ttl files) and example SPARQL queries of that data (.rq format).  These files comprise a small knowledge graph of Zoning information.  Being small it should give flexibility on the software that can be used to query it.

These file queries work with [Apache Jena](https://jena.apache.org/), which has a command line utility `arq` for working with SPARQL queries.

Working example commands are in some of the .rq files, and there are more than what are listed.  The appropriate command line arguments should be in the header of the file.  This was used as a testing ground to make sure that the TTL file could be queried by any SPARQL implementation.

## Most current files
These are the files most relevant to the Zoning KGQAS system and question_templates.py
* [bulk2.ttl](bulk2.ttl) -- Turtle file for "bulk" dimensional requirements.  This uses CDT for distinguishing units.
* [combined.ttl](combined.ttl) -- This file combined `bulk2.ttl` and `permits_use2.ttl` and made a complete knowledge graph.  There were some assumption in SPARQL queries in `generate_template.py` that still requires `bulk2.ttl`.
* [generated_template.py](generated_template.py) -- generates questions from the Knowledge Graph useful for training and testing NLP systems.

## Permitted Uses
* simpl.ttl -- simplest working Turtle File for permitted use.
* multi.ttl -- this adds the complexity of a single district.
* permits_use2.ttl -- this contains most of the permitted uses listed in the International Zoning Code.

* permitted.rq -- simplest query of permitted uses
* permitted2.rq -- This version queries a permitted use and the Zoning District's label.
* all_permitted.rq -- list all the zoning districts with permitted uses.


## Dimensional "Bulk" Requirements 
* bulk.ttl -- Turtle file for "bulk" dimensional requirements.  This version is unitless.
* bulk3.ttl -- Turtle file for "bulk" dimensional requirements.  This uses QUDT for denoting units. (Too complicated for this project).

* bulk.rq -- This is a query for the "bulk" requirements for the maximum Density
* minwidth.rq -- Query for the minimum width requirement for bulk.ttl and bulk2.ttl. 
* rand.rq -- Demonstration of SPARQL RAND() function to add randomization to the returned results.  The paper mentioned using this for training.  Reproducible order is nice so that you can tell if a change improved the method.  rdflib/SPARQL does not seem to have a documented way to seed the random function.  Python code was used to do this instead.

### Notes
* [rdflib_10_questions.ipynb](rdflib_10_questions.ipynb) - The idea was to try to create 10 questions with its associated SPARQL query.  This was early development.  It lead to the development of `generated_template.py`.

* [Custom Datatypes (CDT)](https://ci.mines-stetienne.fr/lindt/v4/custom_datatypes) is one such way of denoting units in Knowledge Graphs.  One problem with it is the values are not numeric, so using isNumeric() in SPARQL will not work.  Instead, QUDT adds more triples to represent the units.

