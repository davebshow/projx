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


=======================================
projx - Graph transformations in Python
=======================================

**Alpha Version API breaking changes may occur over the next several months.**  

projx provides two ways to interact with graphs: 
- A DSL based on Neo4j Cypher for executing graph transformations using a Python DB like API. 
- A programmatic api that consumes JSON ETL configuration objects and executes graph transformations. Based on orientdb-etl.  

Currently only supports networkx.Graph  

- Example Notebook: http://nbviewer.ipython.org/github/davebshow/projx/blob/master/projx_demo.ipynb   

- Real docs coming soon at: http://projx.readthedocs.org/en/latest/#

Thanks to @versae for inspiring this project.