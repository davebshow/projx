# -*- coding: utf-8 -*-
import nxetl


# ETL can be extended beyond NetworkX.
def execute_etl(etl, graph):
    """
    Main API function. Executes ETL on graph.

    :param etl: ETL JSON.
    :param graph: The source graph.
    :return graph: The projected graph.
    """
    etl = _ETL(etl)
    # Extractor is a function that returns a projector class.
    extractor = etl.extractor
    # List of transformers.
    transformers = etl.transformers
    # Loader can be a class or function that contains transformer.
    loader = etl.loader
    # Loader should accept a transfomer list, a projector object, and the
    # graph.
    graph = loader(transformers, extractor, graph)
    return graph


class _ETL(object):

    def __init__(self, etl):
        """
        A helper class that parses the ETL JSON, and initializes the
        required extractor, transformers, and loader. Also store key
        variables used in transformation/loading.

        :param etl: ETL JSON.
        """
        # Get the extractor info.
        try:
            self._extractor = etl["extractor"]
            self.extractor_name = self._extractor.keys()[0]
        except (KeyError, IndexError):
            raise Exception("Please define valid extractor")

        # Get the transformers.
        self.transformers = etl.get("transformers", [])

        # Get the loader info.
        try:
            self._loader = etl["loader"]
            self.loader_name = self._loader.keys()[0]
        except (KeyError, IndexError):
            raise Exception("Please define valid loader.")

        # Get the extractor function.
        self._extractors = {}
        self._init_extractors()
        self.extractor = self.extractors[self.extractor_name]

        # Get the loader function.
        self._loaders = {}
        self._init_loaders()
        self.loader = self.loaders[self.loader_name]

    def _get_extractors(self):
        return self._extractors
    extractors = property(fget=_get_extractors)

    def extractors_wrapper(self, extractor):
        """
        :param extractor: Str.
        """
        def wrapper(fn):
            self.extractors[extractor] = fn
        return wrapper

    def _init_extractors(self):
        """
        Update extractors dict to allow for extensible extractor
        functionality. Here we set the query attribute that will be passed to
        extractor match method. Also, each extractor function will return
        a projector class object that defines a match method.
        """
        @self.extractors_wrapper("networkx")
        def get_nx_extractor(graph):
            """
            :param graph: networkx.Graph
            :returns: projx.nx_extractor
            """
            return nxetl.nx_extractor(self._extractor[self.extractor_name],
                                      graph)

    def _get_loader(self):
        return self._loaders
    loaders = property(fget=_get_loader)

    def loaders_wrapper(self, loader):
        def wrapper(fn):
            self.loaders[loader] = fn
        return wrapper

    def _init_loaders(self):
        """
        Update loaders dict to allow for extensible loader functionality.
        Returns a class/function that performs transformation and loads graph.
        """
        @self.loaders_wrapper("networkx")
        def get_nx_loader(transformers, extractor, graph):
            """
            :param tranformers: List of dicts.
            :extractor: function.
            :param graph: networkx.Graph
            :returns: projx.nx_loader
            """
            return nxetl.nx2nx_loader(transformers, extractor, graph)
