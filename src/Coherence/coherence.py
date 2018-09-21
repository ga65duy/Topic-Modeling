import itertools
import os
import numpy as np
from polyglot.mapping import Embedding
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import pickle
from gensim import matutils
from sklearn.cluster import KMeans
from src.definitions import ROOT_DIR
import time


class Coherence(object):
    """
    Calculate the topic coherence for every topic or the average coherence of all topics in a trained topic model.
    This can be used to compare the quality of different topic models trained on the same texts, e.g. to select the
    best parameters.

    For
    This coherence per topic can also be used to rank the topics of a topic model.

    This module partly relies on the gensim library and thus contains some documentation from gensim.
    """

    def __init__(self, coherence_measure, num_top_tokens, language=None):
        """

        :param coherence_measure: Coherence measure to be used. Supported values are: 'u_mass', 'c_v', 'c_uci', 'c_npmi',
        :type coherence_measure: str
        :param num_top_tokens: Number of top tokens to extract from every topic. The terms will be used to determine the coherence of the topics.
        :type num_top_tokens: int
        :param language: Either 'german' or 'english'. It is required when the selected coherence measure is 'embedding_similarities' or 'embedding_variances'
        :type language: str


        """

        if coherence_measure not in ['u_mass', 'embedding_similarities',
                                     'embedding_variances']:
            raise Exception('{} is not a supported coherence measure'.format(coherence_measure))

        self.coherence_measure = coherence_measure
        self.num_top_tokens = num_top_tokens
        self._embeddings = None

        if coherence_measure in ['embedding_similarities', 'embedding_variances']:
            if language is None:
                raise Exception('For word embedding based coherence measures a language has to be provided.'
                                ' Either "german" or "english". ')
            if language == 'german':
                pass
            elif language == 'english':
                self._embeddings = Embedding.from_glove("D:/Bachelorarbeit/Projekte/tm-maria/models/word_embeddings/glove.6B/glove.6B.100d.txt")
            else:
                raise Exception(
                    'Language {} is not supported. Either "german" or "english".'.format(language))

    def get_topic_coherence(self, topic_model, texts=None, per_topic=False):
        """
        Return the coherence value of the topic model.

        :param topic_model: Pre-trained topic model to evaluate.
        :param texts: Texts that were used to train the topic model. Only required if the selected coherence measure is 'u_mass', 'c_v', 'c_uci' and 'c_npmi'.
        :type texts: str
        :param per_topic: If 'True' return the coherence for each topic of the topic model. Otherwise the mean of all topic cohrences is returned.
        :type per_topic: bool
        :return: The coherence value for the whole topic model or per topic.
        :rtype: int or list(int)
        """
        topics = topic_model.get_topics(num_top_tokens=self.num_top_tokens, values=False)

        if self.coherence_measure == 'embedding_similarities':
            return self._get_embedding_similarities(topics, per_topic)
        elif self.coherence_measure == 'embedding_variances':
            return self._get_embedding_variances(topics, per_topic)

    def _get_embedding_similarities(self, topics, per_topic=False):
        similarities = []
        self._embeddings = self._embeddings.normalize_words()
        for topic in topics:
            combs = itertools.combinations(topic, r=2)
            sims = []
            for x, y in combs:
                try:
                    sims.append(1-(self._embeddings.distances(x, [y])))
                except Exception:
                    continue

            mean_sim = np.mean(sims)
            similarities.append(mean_sim)
        if per_topic:
            return similarities
        else:
            return np.mean(similarities)

    def _get_embedding_variances(self, topics, per_topic=False):
        variances = []
        for topic in topics:
            vectors = []
            for token in topic:
                vectors.append(self._embeddings.word_vec(token))
            matrix = np.array(vectors)
            var = KMeans(n_clusters=1).fit(matrix).inertia_
            variances.append(var)
        if per_topic:
            return variances
        else:
            return np.mean(variances)


    def save(self, mname):
        """
        Save the model to disk.

        :param mname: Path to file. The path is appended to 'models/coherence/'
        :type mname: str
        """
        path = os.path.join(ROOT_DIR, 'models/topic_models', mname)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(mname):
        """
        Load a saved topic model.
        :param mname: path to the stored model. The path is appended to 'models/coherence/'
        :type mname: str
        :return: Vectorizer object
        """
        path = os.path.join(ROOT_DIR, 'models/topic_models', mname)
        with open(path, 'rb') as f:
            vec = pickle.load(f)
        return vec


if __name__ == '__main__':
    pass