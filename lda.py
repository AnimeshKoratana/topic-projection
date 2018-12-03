import graph2 as graph

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
from gensim import corpora, models


np.random.seed(2018)
import nltk
nltk.download('wordnet')


class LDA:
    def __init__(self, graph):
        self.graph = graph
        self.stemmer = SnowballStemmer("english")
        print("Preprocessing the dataset")
        preprocessed_documents = [self._preprocess_document(x.text()) for x in graph.nodes()]
        self.dictionary = gensim.corpora.Dictionary(preprocessed_documents)
        self.dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        print(len(self.dictionary))
        bow_corpus = [self.dictionary.doc2bow(doc) for doc in preprocessed_documents]
        tfidf = models.TfidfModel(bow_corpus)
        self.corpus_tfidf = tfidf[bow_corpus]

    def _preprocess_document(self, text):
        # text= text.decode("utf-8")
        # text.replace("[", "")
        # text.replace("]", "")
        lemmatize_stemming = lambda t : self.stemmer.stem(WordNetLemmatizer().lemmatize(t, pos='v'))
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
        return result

    def fit(self, n_topics = 10, passes = 2, workers = 2):
        self.lda_model = gensim.models.LdaMulticore(self.corpus_tfidf, num_topics=n_topics, id2word=self.dictionary, passes=passes, workers=workers)

def main():
    g = graph.Graph()
    l = LDA(g)
    l.fit(n_topics=100, workers=8, passes=10)
    for idx, topic in l.lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

if __name__ == '__main__':
    main()
