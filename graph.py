import h5py

class Page():
    def __init__(self, index, graph):
        self.idx = index
        self.graph = graph

        self._text = None
        self._title = None
        self._children = None
        self._embedding = None

        self.mean = None

    def text(self):
        if self._text is None:
            self._text = self.graph.get_article_text(self.idx)
        return self._text

    def title(self):
        if self._title is None:
            print("here")
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


class Graph():
    def __init__(self):
        # The document data
        self.w = h5py.File("wiki.hdf5", 'r')
        self.num_pages = 5946517

        # Precalculated Embeddings
        ft = h5py.File("wiki_emb.hdf5", 'r')
        self.e = {}
        self.e['emb'] = ft['emb'].value
        if 'mask' in ft:
            self.e['mask'] = ft['mask'].value
        else:
            self.e = h5py.File("wiki_emb.hdf5", 'r')

    def __len__(self):
        return self.num_pages

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        pg = Page(self.n, self)
        self.n += 1
        return pg

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
