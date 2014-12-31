.. ProjX documentation master file, created by
   sphinx-quickstart on Fri Nov  7 17:00:40 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ProjX's documentation!
=================================

Contents:

.. toctree::
   :maxdepth: 2



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


=====
ProjX
=====

**Pre-Alpha Version API breaking changes may occur over the next several months.**

ProjX is an implementation of a query/transformation DSL for NetworkX. It borrows some syntactic conventions from Neo4j Cypher, but is much simpler, and focuses on matching subgraphs then making transformations to the graph based on node types. Currently, each statement (beggining with verb) iterates over all matched nodes, possibly in the future this will occur in a pipeline to speed things up a bit. Also, the grammar will eventually probably do more validation and create an ETL style JSON (based on Orientdb-ETL maybe) to allow for varied access to the core methods with either JSON or the DSL: 

- Example Notebook: http://nbviewer.ipython.org/github/davebshow/projx/blob/master/projx_demo.ipynb   


The ProjX syntax is as follows:
-------------------------------

Keywords:
=========

Verbs:
------
- "MATCH" Matches a pattern of nodes based on type.
- "TRANSFER" Merges the edges and/or (depending on object) attributes of nodes of one type across a specified sequence of neighboring nodes to nodes of another type.
- "PROJECT" Projects a relationship between nodes of one type across a specified sequence of neighboring nodes.
- "COMBINE" Combines two node types creating new nodes.
- "RETURN" Specify table/graph and nodes to return. NOT IMPLEMENTED

Objects:
--------
Objects act as parameters that can be passed to verbs.

- "GRAPH" When used with "MATCH" returns whole graph to operated over, not just a subgraph.
- "SUBGRAPH" When used with "MATCH", results in just a matched subgraph, this is the default setting, so is not necessary except for transparency.
- "ATTRS" When used with "TRANSFER" only attributes will be transfered.
- "EDGES" When used with "TRANSFER" only edges will be transfered.
- "TABLE" Not implemented.

Predicates:
-----------

Predicates allow for fine tuned control of graph operations.

- "DELETE" Determines what nodes should be deleted after performing verb based operations.
- "METHOD" Describes method for determining edge weight upon projection, works when edges are projected over a single intermediate node type. Only implemented for Jaccard similarity coefficient.
- "SET" Determines what attributes will be retained and where after a "TRANSFER" or "PROJECT" statement.
- "WHERE" NOT IMPLEMENTED
- "NEW" A generic command that represent the new edges created by a "PROJECT" or the new nodes created by "COMBINE". 

Patterns:
=========

Nodes:
------
Nodes are represented using (). Include an alias with the (): (t1:Type1). This allows for
cleaner code and prevents errors when using complex pattern
that repeat types.
- (f:Foo)
- (b:Bar)

Edges:
------
Edges are represented using []: [e1:EdgeType1]

Patterns:
---------
A pattern is a combination of nodes and edges. It creates a
"type sequence", or a set of criteria that determain a legal
path during graph traversal based on node's type_attr. For
example, if we want to locate all nodes with type_attr == 'Type1'
that are connected to nodes with type_attr == 'Type2', the pattern
would be specified as "(t1:Type1)-(t2:Type2)". A pattern can be as
long as necessary, and can repeat elements. Note that the traversal
does not permit cycles. Also, we can create patterns for edge traversal,
specifying edge types.
- "(f1:Foo)-(b:Bar)-(f2:Foo)"
- "(d:Dog)-[b:bites]-(c:Cat)"

Thanks to @versae for inspiring this project.