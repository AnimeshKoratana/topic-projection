import h5py
import torch
import numpy as np
from queue import Queue

class Page():
    def __init__(self, index, graph):
        self.idx = index
        self.graph = graph

        self._text = None
        self._title = None
        self._children = None
        self._embedding = None

        self.mean = None

    def __eq__(self, x):
        return self.idx == x.idx

    def text(self):
        if self._text is None:
            self._text = self.graph.get_article_text(self.idx)
        return self._text

    def title(self):
        if self._title is None:
            self._title = self.graph.get_article_title(self.idx)
        return self._title

    def children(self):
        if self._children is None:
            self._children = [Page(i, self.graph) for i in self.graph.get_article_links(self.idx)]
        return self._children

    def embedding(self):
        if self._embedding is None:
            self._embedding = self.graph.get_article_emb(self.idx)
        return self._embedding
        # return torch.from_numpy(self._embedding)


class Graph():
    def __init__(self, seed = 2586555, num_pages= 1000):
        self.max_pages = 5946517
        self.num_pages = num_pages
        if seed is None:
            seed = np.random.randint(0, self.max_pages)
            print("Randomly Chose Seed: ", seed)
        self.seed = seed % self.max_pages

        self.w = h5py.File("wiki.hdf5", 'r')

        # Precalculated Embeddings
        ft = h5py.File("wiki_emb.hdf5", 'r')
        self.e = {}
        self.e['emb'] = ft['emb'].value
        if 'mask' in ft:
            self.e['mask'] = ft['mask'].value
        else:
            self.e = h5py.File("wiki_emb.hdf5", 'r')

        self.subgraph = self._init_graph(self.seed)

    def _init_graph(self, seed):
        subgraph = []

        children = Queue()
        children_idx = [seed]
        children.put(Page(seed, self))
        while len(subgraph) < self.num_pages:
            node = children.get()
            subgraph.append(node)
            for c in node.children():
                if c.idx not in children_idx:
                    children.put(c)
                    children_idx.append(c.idx)

        return subgraph

    def __len__(self):
        return self.num_pages

    def __getitem__(self, key):
        return self.subgraph[key]

    def __iter__(self):
        self.iter_subgraph = iter(self.subgraph)
        return self

    def __next__(self):
        return next(self.iter_subgraph)

    def get_article_title(self, article_id):
        return self.w['title'][article_id]

    def get_article_text(self, article_id):
        return self.w['text'][article_id]

    def get_article_emb(self, article_id):
        return self.e['emb'][article_id]

    def get_title_iter(self):
        return self.w['title']

    def get_article_links(self, article_id):
        links = self.w["links".encode()][article_id].strip().decode().split(' ')
        if links[0] != '':
            links = [int(i) for i in links]
        else:
            links = []
        return links
