# Getting started

**A tour of projx**

**projx transforms** graphs. It takes a source graph as input, matches a traversal pattern, and generates a stream of data that is transformed and loaded into a NetworkX graph in memory or written to a persistent data store. **projx** focuses on producing graph wide transformations that operate over nodes and edges matching a traversal pattern in a multipartite graph. This concept can be concisely demonstrated using the DSL to transform NetworkX graphs in memory: **[check out this demo IPython Notebook](http://bit.ly/1EiMaMt)** to see **projx** in action.

## Installation

**projx** is available through [PyPi](https://pypi.python.org/pypi/projx), and can be installed with pip:

```bash
$ pip install projx
```

or feel free to fork or clone [projx on github](https://github.com/davebshow/projx)

```bash
$ git clone https://github.com/davebshow/projx.git
```
## Using projx with NetworkX

### The DSL

The **projx DSL** is a declarative query language based on Neo4j's Cypher. It consists of MATCH statements and transformations. To use the it, first instantiate a projection of a multipartite NetworkX graph:

```python
import projx as px

# This returns a multipartite networkx.Graph where each node has an
# attribute "type".
graph = test_graph()
p = px.Projection(graph)
```

#### The match clause
Then execute a MATCH statement written in the projx [DSL](dsl.md):

```python
# This returns an instance of networkx.Graph
p.execute("MATCH (m)-(n)")
```

The above match pattern will match all dyads (two nodes connected by an edge) in the graph. Notice there is no RETURN statement, the projx.Projection.execute builds and returns an instance of networkx.Graph that contains all paths matched by the match pattern.

To match a subgraph of the original graph, we can limit our match pattern using criteria such as node type to only match a subset of paths. The following statement matches a path of all people who are connected through a city, and then returns a bipartite graph of people and the cities they are connected to:

```python
# This returns an instance of networkx.Graph
p.execute("MATCH (p1:Person)-(c:City)-(p2:Person)")
```

Notice the syntax for denoting a node is quite simple, it consists of parenthesis containing an alias, a colon delimiter, and then a (p2:Person)node type (alias:NodeType).

Edge type criteria can also be used in pattern matching. Observe:

```python
# This returns an instance of networkx.Graph
p.execute("MATCH (p1:Person)-[l:lives_in]-(c:City)")
```

This statement only matches limits matching people and cities by the type of edge connecting them, in this case "lives in".

See the [DSL](dsl.md) docs for a full description of match.

#### Transformation clauses

After we match a pattern, we would often like to transform it in some way or another. Taking the bipartite graph of types Person and City from above as an example, we may want to compress it into a one mode graph like a social network. **projx** makes this easy:

```python
# This returns a one mode social network of people who are associated through
# nodes of type city.
p.execute("""
    MATCH (p1:Person)-(c:City)-(p2:Person)
    PROJECT (p1)-(p2)
    DELETE c
""")
```

A transformation clause, in this case, MATCH takes a pattern similar to the match pattern as an argument. However, unlike the match pattern, the transformation pattern employs only the alias established by the match patter.

Furthermore, we can specify attributes that we would like to set on the newly created elements (in the case of PROJECT, a new edge), nodes we would like to delete from the projection referenced simply through their alias, and special methods. The following demonstrates using a special method to make an edge weight calculation during the projection using the Newman technique:

```python
p.execute("""
    MATCH   (p1:Person)-(wild)-(p2:Person)
    PROJECT (p1)-(p2)
    METHOD JACCARD Institution, City
    SET     name = wild.label
    DELETE  wild
""")
```

Notice the method NEWMAN takes node types as arguments. These determine what sort of connections between people will be used in the edge weight calculation.

The projx DSL also implements the transformations TRANSFER and COMBINE. For a full description of transformations and the DSL, please refer to the [DSL docs](dsl.md)

### ETL

When you run the DSL, the first thing projx does is parse the query, producing a JSON structure that is the **projx** version of an ETL config file. This concept is based on orientdb-etl. The ETL is a simply a JSON config file or Python dict data structure. Here's an example of what the parser returns:

```python
print(json.dumps(px.parse_query(("""
    MATCH   (p1:Person)-(wild)-(p2:Person)
    PROJECT (p1)-(p2)
    METHOD JACCARD Institution, City
    SET     name = wild.label
    DELETE  wild
""")), indent=2))

{
  "extractor": {
    "networkx": {
      "type": "subgraph",
      "traversal": [
        {
          "node": {
            "alias": "p1",
            "type": "Person"
          }
        },
        {
          "edge": {}
        },
        {
          "node": {
            "alias": "wild"
          }
        },
        {
          "edge": {}
        },
        {
          "node": {
            "alias": "p2",
            "type": "Person"
          }
        }
      ]
    }
  },
  "transformers": [
    {
      "project": {
        "pattern": [
          {
            "node": {
              "alias": "p1"
            }
          },
          {
            "edge": {}
          },
          {
            "node": {
              "alias": "p2"
            }
          }
        ],
        "set": [
          {
            "value_lookup": "wild.label",
            "key": "name"
          }
        ],
        "method": {
          "jaccard": {
            "args": [
              "Institution",
              "City"
            ]
          }
        },
        "delete": {
          "alias": [
            "wild"
          ]
        }
      }
    }
  ],
  "loader": {
    "nx2nx": {}
  }
}
```

This structure, which will be thoroughly addressed in the [next section](#using-the-etl-api), is then simply passed to the other main API function that executes the ETL pipeline:

```python
etl = px.parse_query("""
    MATCH   (p1:Person)-(wild)-(p2:Person)
    PROJECT (p1)-(p2)
    METHOD JACCARD Institution, City
    SET     name = wild.label
    DELETE  wild
""")
# Main API function.
subgraph = px.execute_etl(etl, g)
```

The following sections will detail the ETL, explaining how it is used with NetworkX **and** how it can be used to translate graph data to and from various data sources.

## Using the ETL API

The **ETL API** is simply a JSON configuration object passed to the ETL pipeline as show above. It has the advantage of being extremely simple, it requires little or no string construction and is easy to build programmatically. When transferring data between databases or flat files, this kind of configuration can be easily stored as a file and passed as a command line argument.

At the same time, this type of object in infinitely extensible in that the developer can pass any arbitrary key/value pair necessary for custom components without disrupting other elements of the pipeline. Transformers are defined purely as JSON objects, while the rules for their parsing and execution are defined in the loader function. For a more complete explanation of the **projx ETL** see the [Extending the ETL section](extending-etl.md).

The ETL JSON consists of three objects: an extractor, transformers, and a loader. The extractor gets the necessary data to open a data stream from the data source and apply the transformations. The transformations are applied in a pipeline to the data stream generated by the extractor. The loader defines to where and how it will be loaded. Let's look at a couple examples of ETL configurations, one element at a time:

### Example: NetworkX -> NetworkX: nx2nx

Here we'll dissect a NetworkX to NetworkX ETL JSON configuration object. It all begins with an extractor:

```python
"extractor": {
    "networkx": {
        "type": "graph",
        "node_type_attr": "type",
        "edge_type_attr": "type",
        "traversal": [
            {"node": {"alias": "c", "type": "City"}},
            {"edge": {}},
            {"node": {"alias": "i", "type": "Institution"}}
        ]
    }
}
```

This object provides the following information:

* The type of extractor to be used
* The type of projection (graph or subgraph)
* The name used to define the node type in the source graph
* The name used to define the node type in the source graph
* A traversal which defines the match pattern that will be executed on the source graph: all nodes of type "City" connected to nodes of type "Institution".

Next, we define a list of transformations that will be performed upon the nodes returned by the match pattern:

```python
"transformers": [
    {"transfer": {
            "pattern": [
                {"node": {"alias": "c"}},
                {"edge": {}},
                {"node": {"alias": "i"}}
            ],
            "set": [
                {"key": "city", "value_lookup": "c.label"}
            ],
            "method": {
                "edges": {
                    "args": ["Person"]
                }
            },
            "delete": {
                "alias": ["c"]
            }
        }
    }
],
```

This object provides a list of transformer that will be parsed by the loader. It includes:

* The name of the transformer
* The pattern, corresponding to the alias created in the extract object, which specifies the nodes on which the transformation will be performed
* The attribute key to be set, and the lookup based on node alias to populate the attribute values
* The method to be used in the transformation, in this case edges from nodes of type "Person" will be transfered to the destination nodes with alias "i"
* The nodes to be deleted, a simple list of node alias

Finally, the loader object:

```python
"loader": {
    "nx2nx": {}
}
```

Nothing to it. Just specifies the name of the loader we will use. Notice that the loader specifies both the source and the target data "nx2nx". This is because, while extractors are specific only to their data source, loaders are the glue between the source and target, and are therefore couple to both. We will see this again in the following example, neo4j2nx.

### Example: Neo4j -> NetworkX: neo4j2nx

Now let's look at the ETL for a Neo4j to NetworkX transformation. We'll use a database loaded with smoothie recipes. It is a simple bipartite graph in which nodes of type Ingredient are connect to nodes of type Recipe. Examine the extractor:

```python
"extractor": {
    "neo4j": {
        "query": "match (n:Ingredient)--(r:Recipe)--(m:Ingredient) return n, r, m",
        "source": "http://localhost:7474/db/data/"
    }
}
```

The following information is provided:

* The type of extractor (neo4j)
* The query to be executed on the database.
* The url required to connect to a running instance of Neo4j

The transformers are equally simple. They simply map this data to a NetworkX graph held in memory. The following are examples of transformers used with the neo4j2nx_loader:

```python
"transformers": [
    {"node": {
        "pattern": [{"node": {"alias": "n", "unique": "UniqueId"}}],
        "set": [
            {"key": "name", "value_lookup": "n.UniqueId"},
            {"key": "type", "value": "Ingredient"}
        ]
    }}

    {"edge": {
        "pattern": [
            {"node": {"alias": "n", "unique": "UniqueId"}},
            {"edge": {}},
            {"node": {"alias": "m", "unique": "UniqueId"}}
        ],
        "set": [
            {"key": "name", "value_lookup": "r.UniqueId"}
        ],
    }}
]

```

Like the nx2nx transformers, they provide the following information:

* The transformer keyword (node or edge)

* A pattern: a list of nodes and edges representing the pattern used in the transformer. Notice with neo4j2nx we have to specify the attribute representing the unique id that will be used as the NetworkX node index in graph creation

* Attributes to be set on the newly created node or edge

Finally, the loader is exceedingly simple, just declaring the name of the loader to be used:

```python
"loader": {
    "neo4j2nx": {}
}
```

Check out this [demo notebook](http://nbviewer.ipython.org/github/davebshow/projx/blob/master/projx_neo4j_demo.ipynb) for some more examples of neo4j2nx.

### Phew! Still want more? In the [next section](extending-etl.md), we pick apart the ETL pipeline and learn how to extend it with custom components.
