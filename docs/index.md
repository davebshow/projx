# projx - Graph transformations in Python.

**alpha**

**projx** provides a simple and extensible API for interacting with graphs in Python. Its core functionality is built around making graph transformations using the [NetworkX](https://networkx.github.io/) module and a DSL based on [Neo4j's](http://neo4j.com/) [Cypher](http://neo4j.com/docs/stable/cypher-query-lang.html) query language. It also provides an extensible ETL pipeline that uses JSON configuration (roughly modeled after [orientdb-etl](https://github.com/orientechnologies/orientdb-etl/wiki)) to translate graph data between various persistent and in-memory representations.


- [Getting Started](getting-started.md)
    - [Installation](getting-started.md#installation)
    - [Using projx with NetworkX](getting-started.md#using-projx-with-networkx)
    - [Using the ETL API](getting-started.md#using-the-etl-api)



**Some other relevant links:**

projx on [PyPI](https://pypi.python.org/pypi/projx)

Demo Notebook with [projx DSL](http://bit.ly/1EiMaMt)

Demo Notebook with [Neo4j2NetworkX](http://nbviewer.ipython.org/github/davebshow/projx/blob/master/projx_neo4j_demo.ipynb)

Demo Notebook: [Loading the Flickr group graph to Neo4j](http://nbviewer.ipython.org/github/davebshow/projx/blob/master/flicker_graph.ipynb)


Thanks to [@versae](https://github.com/versae) for inspiring this project.
