import graph

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
from gensim import corpora, models

from scipy.special import kl_div as kl


np.random.seed(2018)
import nltk
nltk.download('wordnet')


class LDA:
    def __init__(self, pages, eval_percent=0):
        cutoff = int(len(pages) - (len(pages) * (eval_percent/100)))
        self.pages = pages
        self.stemmer = SnowballStemmer("english")
        print("Preprocessing the dataset")
        preprocessed_documents = [self._preprocess_document(x.text()) for x in pages]
        self.dictionary = gensim.corpora.Dictionary(preprocessed_documents)
        self.dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        print(len(self.dictionary))
        bow_corpus = [self.dictionary.doc2bow(doc) for doc in preprocessed_documents[:cutoff]]
        tfidf = models.TfidfModel(bow_corpus)
        self.corpus_tfidf = tfidf[bow_corpus]
        self.lda_model = None

        val_corpus = [self.dictionary.doc2bow(doc) for doc in preprocessed_documents[cutoff+1:]]
        val_tfidf = models.TfidfModel(val_corpus)
        self.corpus_tfidf_val = val_tfidf[val_corpus]

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

    def compare(self, other):
        # Compare our lda model versus another one using the average Jenson Shannon Divergence between each of the topics
        assert(isinstance(other, type(self)))
        divergences = []
        for ti in range(self.lda_model.num_topics):
            for tj in range(other.lda_model.num_topics):
                topic_i = self.lda_model.state.get_lambda()[ti]
                topic_i = topic_i / topic_i.sum()
                topic_j = other.lda_model.state.get_lambda()[tj]
                topic_j = topic_j / topic_j.sum()
                M = 0.5 * (topic_i + topic_j)
                jsd = 0.5 * kl(topic_i, M) + 0.5 * kl(topic_j, M)
                divergences.append(jsd)

        return np.mean(np.array(divergences))

    def __call__(self, x):
        return self.infer(x)

    def fit(self, n_topics = 10, passes = 20, workers = 8):
        self.lda_model = gensim.models.LdaMulticore(self.corpus_tfidf, num_topics=n_topics, id2word=self.dictionary, passes=passes, workers=workers)

    def infer(self, text):
        # first make bow vector of text
        # returns a [(topic_id, p(topic)) ...]
        preprocessed = self._preprocess_document(text)
        bow_vector = self.dictionary.doc2bow(preprocessed)
        distribution = self.lda_model.get_document_topics(bow_vector)
        return distribution

    def average_perplexity(self):
        return np.mean(self.lda_model.log_perplexity(self.corpus_tfidf_val))

# def main():
#     g = graph.Graph()
#     l = LDA(g.nodes())
#     l.fit(n_topics=100, workers=8, passes=10)
#     for idx, topic in l.lda_model.print_topics(-1):
#         print('Topic: {} \nWords: {}'.format(idx, topic))
#
# if __name__ == '__main__':
#     main()
