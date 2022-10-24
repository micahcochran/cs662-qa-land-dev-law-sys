# Description of files

These are files that make up a triples "Database" using Turtle RDF (.ttl files) and example SPARQL queries of that data (.rq format).  These files comprise a small knowledge graph.  Being small it should give flexibility on the software that can be used to query it.

These file queries work with Apache Jena. This is Java and may not work so well with Python.
https://jena.apache.org/

Working example commands are in some of the .rq files.  The command is 'arq', which is the command for Jena to do a SPARQL query.

## Permitted Uses
* simpl.ttl -- simplest working Turtle File for permitted use.
* multi.ttl -- this adds the complexity of a single district.
* permits_use2.ttl -- this contains most of the permitted uses listed in the International Zoning Code.

* permitted.rq -- simplest query of permitted uses
* permitted2.rq -- This version queries a permitted use and the Zoning District's label.
* all_permitted.rq -- list all the zoning districts with permitted uses.


## Bulk Requirements 
* bulk.ttl -- These are "bulk" dimensional requirements.  This version has no units.
* bulk2.ttl -- These are "bulk" dimensional requirements.  This uses CDT for distinguishing units.

* bulk.rq -- This is a query for the "bulk" requirements for the maximum Density
* minwidth.rq -- Query for the minimum width requirement for bulk.ttl and bulk2.ttl. 
