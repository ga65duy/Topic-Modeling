from sklearn.decomposition import NMF
import sklearn as sk
from gensim.models import LdaModel, CoherenceModel
from gensim.models.callbacks import PerplexityMetric
from gensim.corpora.dictionary import Dictionary
from src.definitions import ROOT_DIR
import pandas as pd
from gensim import matutils
import numpy as np
import pickle
import os
import logging
import sys
import io


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model_library_mapping = {'nmf': 'sklearn',  'lda': 'gensim'}


class TopicModel(object):
    """
    Train topic models based on the NMF implementation of sklearn and LDA implementation of Gensim.
    It also contains methods to analyze and visualize the learned topic-term matrix and coherence
    measures to analyze the quality of the topic model.

    """
    # Initialization and directly start training
    def __init__(self, model_name, num_topics, train_document_term_matrix, id2token, test_document_term_matrix=None, **kwargs):
        """


        :param model_name: The topic model to initialize. Supported values are 'NMF' and 'LDA'
        :type model_name: str
        :param num_topics: The  number of topics
        :type num_topics: int
        :param train_document_term_matrix: Training data for the model
        :type train_document_term_matrix: np.ndarray
        :param id2token: Mapping of ids to token strings.
        :type id2token: dict(int, str)
        :param test_document_term_matrix: If a held out test set is passed, it is used to calculate the perplexity during training of LDA models.
        Otherwise, the perplexity is evaluated on the training set.
        :type test_document_term_matrix: np.ndarray
        :param kwargs: Arguments passed to sklearn or gensim topic model implementations

        The NMF method is optimizing the Frobenius norm of the original matrix minus the product of the document-topic
        and topic-token matrices. The matricies W and H are initialized with NNDSVDa and
        uses the multiplicative update method.

        For further parameters of LDA please visit the respective
        gensim documentation. The arguments listed there can be passed when
        initialzing.
        """

        self.model_name = model_name.lower()
        self.num_topics = num_topics
        self.train_document_term_matrix = train_document_term_matrix
        self.id2token = id2token
        # During training calculate perplexity for LDA and reconstruction error for nmf
        self._convergence = None
        self.trained = False


        try:
            if model_library_mapping[model_name] == 'sklearn':
                self.model = self._init_sklearn_model(model_name, num_topics)
                self._train_sklearn_model(train_document_term_matrix)
                self._topic_term_matrix = self.model.components_

            elif model_library_mapping[model_name] == 'gensim':
                self.model = self._init_gensim_model(model_name, num_topics, train_document_term_matrix, id2token, test_document_term_matrix, **kwargs)
                self._topic_term_matrix = self.model.get_topics()
                self._convergence = self.model.metrics['perplexity']
        except KeyError:
            raise Exception('{} not supported'.format(model_name))

    # Methods for Gensim Models
    def _init_gensim_model(self, model, num_topics, train_document_token_matrix, id2token, test_document_token_matrix=None, **kwargs):
        # Gensim

        if test_document_token_matrix is not None:
            corpus = matutils.Sparse2Corpus(test_document_token_matrix, documents_columns=False)
        else:
            corpus = matutils.Sparse2Corpus(train_document_token_matrix, documents_columns=False)
        pm = PerplexityMetric(corpus, logger='shell', title='perplexity')

        corpus = matutils.Sparse2Corpus(train_document_token_matrix, documents_columns=False)
        if model == 'lda':
            return LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2token, callbacks=[pm], eval_every=None,  **kwargs)

    def _inference_gensim(self, document_token_matrix):
        corpus = matutils.Sparse2Corpus(document_token_matrix, documents_columns=False)
        document_topic_matrix, n = self.model.inference(corpus)
        return document_topic_matrix

    # Methods for Sklearn Models
    def _init_sklearn_model(self, model, num_topics):
        # Sklearn supports nmf
        if model == 'nmf':
            return NMF(n_components=num_topics, solver='mu', verbose=True, init='nndsvda', max_iter=200, tol=1e-9)



    def _train_sklearn_model(self, document_token_matrix):
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()

        self.model.fit(document_token_matrix)

        sys.stdout = old_stdout
        errors_output = mystdout.getvalue()
        errors_list = []
        for line in errors_output.split('\n'):
            print(line)
            tokens = line.split()
            if len(tokens) > 0:
                errors_list.append(tokens[-1])
        self._convergence = [float(error) for error in errors_list]

    def _transform_sklearn(self, document_token_matrix):
        document_topic_matrix = self.model.transform(document_token_matrix)
        return document_topic_matrix


    def get_topic_token_matrix(self, normalize=True):
        """
        Return the learned topic token matrix.

        """
        topic_term_matrix = self._topic_term_matrix
        if normalize:
            topic_term_matrix = sk.preprocessing.normalize(topic_term_matrix, norm='l1')
        return topic_term_matrix

    def get_document_topic_matrix(self, document_token_matrix, normalized=True):
        """
        Apply the topic model on the passed document_token_matrix and return the document_topic_matrix.
        """

        if model_library_mapping[self.model_name] == 'sklearn':
            document_topic_matrix = self._transform_sklearn(document_token_matrix)
        else:
            document_topic_matrix = self._inference_gensim(document_token_matrix)

        if normalized:
            document_topic_matrix = sk.preprocessing.normalize(document_topic_matrix, norm='l1')
        return document_topic_matrix

    def get_training_convergence(self):
        return self._convergence

    # Methods to persist and load the trained models.
    def save(self, path):
        """
        Save the model to disk.

        :param path: Path to file. The path is appended to 'models/topic_models/'
        :type path: str
        """
        path = os.path.join(ROOT_DIR, 'models',  path)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        Load a saved topic model.
        :param mname: path to the stored model. The path is appended to 'models/topic_models/'
        :type mname: str
        :return: Vectorizer object
        """
        path = os.path.join(ROOT_DIR, 'models', path)
        with open(path, 'rb') as f:
            vec = pickle.load(f)
        return vec

    def get_topics(self, num_top_tokens=10, values=False):
        """
        Return the N top tokens for every topic.

        :param num_top_tokens: Number of tokens to return for
        :param topic_ids:
        :param values:
        :return:
        """
        top_topics = []
        ttm = self.get_topic_token_matrix()
        num_topics, num_tokens = ttm.shape

        if num_top_tokens > num_tokens:
            num_top_tokens = num_tokens

        for topic in ttm:
            sorted_ids = matutils.argsort(topic, num_top_tokens, reverse=True)
            if values:
                sorted_tokens = [(self.id2token[idx], topic[idx]) for idx in sorted_ids]
            else:
                sorted_tokens = [self.id2token[idx] for idx in sorted_ids]
            top_topics.append(sorted_tokens)
        return top_topics

    def get_topics_dataframe(self, num_top_tokens=10, values=False):
        """
         Return the num_top_tokens per topic as a Pandas DataFrame.
        :param num_top_tokens:
        :param values: If true a tuple of (term, probability) is stored in the DataFrame.
        :return: pd.DataFrame
        """
        topics = self.get_topics(num_top_tokens, values)
        row_labels = ['Topic {}'.format(i) for i in range(0, len(topics))]
        df = pd.DataFrame(topics, index=row_labels)
        return df

    def get_topics_ranked(self, ranking, num_top_tokens=10, ascending=False, values=False):
        """
        Rank the topics by ranking and return the num_top_tokens each.

        :param ranking: A value for every topic as a list. The lenght of the list must equal the number of topics.
        :param ascending: If True the topics will br sorted ascending
        :param values: If true a tuple of (term, probability) is stored in the DataFrame.
        :return: pd.DataFrame
        """
        topic_df = self.get_topics_dataframe(num_top_tokens, values)
        assert(len(ranking) == len(topic_df.index))

        dfn = topic_df.copy()
        dfn.insert(len(dfn.columns), 'score', ranking)
        dfn = dfn.sort_values(by=['score'], ascending=ascending)
        dfn = dfn.reindex(columns=(['score'] + list([a for a in dfn.columns if a != 'score'])))
        return dfn


    def get_topic_coherence(self, num_top_tokens=10, per_topic=False):
        """
        Return the topic coherence for the trained measures. The coherence is a measure that indicates how
        human interpretable the captured topics are.
        For all combinations of the num_top_tokens per topic (by probability) the similarity is calculated and averaged for the topic.
        The coherence of a topic model is the mean of the coherence values for each topic.
        The 'u_mass' measure was introduced by Mimno et. al. (2011) and utilizes
        solely the word co-occurences in the training documents.
        :param num_top_tokens: The number of top terms to extract from each topic.
        :type num_top_tokens: int
        :param per_topic: If true a list of coherence values for each topic is returned. Passing this list to
        :py:func:`get_topics_ranked` results in the topics being ranked by their coherence score
        :type per_topic: bool
        :rtype: int or list(int)

        """

        corpus = matutils.Sparse2Corpus(self.train_document_term_matrix, documents_columns=False)
        dictionary = Dictionary.from_corpus(corpus, self.id2token)
        topics = self.get_topics(num_top_tokens)
        u_mass = CoherenceModel(coherence='u_mass', topics=topics, corpus=corpus, dictionary=dictionary)
        if per_topic:
            return u_mass.get_coherence_per_topic()
        else:
            return u_mass.get_coherence()


if __name__ == '__main__':
    # import src.visualization.visualize_tm as vis
    # model = TopicModel.load('test-nmf-save-file.pkl')
    # solution, task = vis.get_word_intrusion_dataframe(model)
    # print(solution.to_csv(sep='\t'))
    # print()
    # print(task.to_csv(sep='\t'))
    pass

