ProjX
=====
The ProjX syntax is as follows:
-------------------------------

Verbs:
======
    - "MATCH" Matches a pattern of nodes based on type.
    - "MATCH PARTIAL" Like "MATCH", but also matchs a partial pattern.
      	Coming soon...
    - "MERGE" Merges the edges and attributes of nodes of one
 		type across a specified sequence of neighboring nodes
 		to nodes of another type.
    - "TRANSFER" Like "MERGE", but only transfers attributes.
    - "PROJECT" Projects a relationship between nodes of one
		type across a specified sequence of neighboring nodes.
    - "RETURN" Specify table/graph and nodes to return. Coming soon...

Patterns:
=========

Nodes:
------
Nodes are represented using (). For the minimal syntax, the
() contains at least a node type specification, this specification
corresponds to the attribute set at init type_attr: (Type1).
For longer queries over subgraphs, it is recommended to
include an alias with the (): (t1:Type1). This allows for
cleaner code and prevents errors when using complex pattern
that repeat types.
    - (f:Foo)
    - (b:Bar)

Edges:
------
Currently, only simple undirected edges are permited. They are
denoted with the hyphen -. Support for edge types and attrs are
coming soon.

Patterns:
---------
A pattern is a combination of nodes and edges. It creates a
"type sequence", or a set of criteria that determain a legal
path during graph traversal based on node's type_attr. For
example, if we want to locate all nodes with type_attr == 'Type1'
that are connected to nodes with type_attr == 'Type2', the pattern
would be specified as "(t1:Type1)-(t2:Type2)". A pattern can be as
long as necessary, and can repeat elements. Note that the traversal
does not permit cycles.
    - "(f1:Foo)-(b:Bar)-(f2:Foo)"
    - "(d:Dog)-(p1:Person)-(p2:Person)-(c:Cat)"

Queries:
========
ProjX queries combine a verb with a pattern to perform some kind
of search or schema modification over the graph. If queries begin
with a "MATCH" clause, they will project across only the matched
subgraph, thus discarding all nodes that do not match. If queries
begin with a different clause, they will still only be able to act
upon the first matched pattern; however, they retain all other nodes
not involved in the pattern regards of other operations.

Matched subgraph queries:
-------------------
Matched subgraph queries must begin with a "MATCH" statement. This
produces the subgraph upon which the rest of the verbs will
operate. After a graph is match, other projections can be perfomed
upon the resulting subgraph. For example, let's imagine we want to
project a For example, let's imagine we want to project a social
network of 'Person' nodes through their association with nodes of
type 'Institution'. First we match the subgraph, and then make
the projection:


'''      
***MATCH (p1:Person)-(i:Institution)-(p2:Person)***  
***PROJECT (p1)-(i)-(p2)***  
'''

In the above example it is important to note the mandatory use of
the alias style node syntax. To go a step further, let's transfer the
edges and atributes contained in nodes of type 'City' to neighboring
nodes of type 'Person', and then project the same social network of
'Person' nodes through their association with nodes of type
'Institution'. The query is as follows:

'''  
***MATCH (c:City)-(p1:Person)-(i:Institution)-(p2:Person)***  
***TRANSFER (c)-(p1)***  
***PROJECT (p1)-(i)-(p2)***  
'''

^ In an undirected graph "TRANSFER (c)-(p1)" finds p2 as well.

And we can keep making up examples:

'''  
***MATCH (p1:Person)-(c:City)-(i:Institution)-(p2:Person)***  
***MERGE (c)-(i)***   
***PROJECT (p1)-(i)-(p2)***  
'''  

'''  
***MATCH (p1:Person)-(i:Institution)-(c:City)-(p2:Person)***     
***TRANSFER (i)-(p1)***    
***TRANSFER (c)-(p2)***    
***MERGE (c)-(i)***  
***PROJECT (p1)-(i)-(p2)***  
...


CURRENTLY ProjX only allows ***1 match per query***! Also, in
the future it will probably allow soft pattern matching to match
partial patterns as well.

Full graph queries:
-----------------
To perform an projection over the whole graph and return a modified
copy, simply tell ProjX what you want to do by combining a verb and
a simple pattern. Node Type aliases are not necessary for one-line
queries UNLESS you are using the same node type multiple times in the
pattern. For example, to transfer all the attributes of nodes of
type 'Foo' to their neighboring nodes of type 'Bar' and delete the
'Foo' nodes we can say:


***"TRANSFER (Foo)-(Bar)"***


We can also use a wildcard node type when we want a more flexible
traversal. Let's project an association graph of people connected
to other people through a node of any other type:

***"PROJECT (p1:Person)-()-(p2:Person)"*** 

Here it is important to remember this projection will delete the
wildcard nodes, but any other nodes that are not matched by this
pattern remain in the graph. If you only want the returned graph
to contain the 'Person' nodes, you will need to use a Multi-line
query as defined in the following section. For a final example, let's
"TRANSFER" 'Institution' attributes to nodes of type 'City', but only
when they are connected through a node of type 'Person':

***"TRANSFER (Institution)-(Person)-(City)"***
 
And we can continue:


We can still write multi-line queries that act over the whole graph
too.

"""  
TRANSFER (i:Institution)-(p:Person)-(c:City)  
MERGE (c)-(p)  
"""


***"MERGE (Foo)-(Bar)"***
...


Note that single line queries use the first and last specified
nodes as the source and target of the operation.

Predicates:
-----------
ProjX doesn't currently support predicates such as "AS", "WHERE",
but it will soon.

:param query: String. A ProjX query.
:returns: networkx.Graph. The graph or subgraph with the required
  schema modfications.