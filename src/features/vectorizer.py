from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.definitions import ROOT_DIR
import pickle
import os


class Vectorizer(object):
    """
    Convert a set of documents to a document-term matrix. It is assumed that the documents are already processed and
    tokenized no further preprocessing steps will be applied. The collection of documents should be a list of strings
    in which every document can be split into tokens by calling .split()

    The implementation relies of sklearns `CountVectorizer <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_ and `TfIdfVectorizer <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>`_. For more information,
    please refer to the respective documentation.
    """

    def __init__(self,  weighting, training_documents, **kwargs):
        """
        :param weighting: Weighting assigned to the terms in the document-term matrix
        :type weighting: 'tf' or 'tfidf'
        :param training_documents: Training documents to build the vector space.
        :type training_documents: list of strings
        :param kwargs: See the documentation of sklearn CountVectorizer and TfIfdVectorizer to find further parameters.
        It is recommended to at least specify the max_n_terms, min_df, and max_df arguments. Mostly these three arguments
        were adapted during the evaluation. Note that a custom tokenizer can not be passed.
        """
        self.weighting = weighting

        # Initilaize sklearn model
        self.sk_model = self._init_model(**kwargs)

        # Train the model
        self.sk_model.fit(training_documents)
        self.token2id_mapping = self.sk_model.vocabulary_

    def _init_model(self, **kwargs):
        if self.weighting == 'tf':
            return CountVectorizer(tokenizer=self._tokenize, **kwargs)
        elif self.weighting == 'tfidf':
            return TfidfVectorizer(tokenizer=self._tokenize, **kwargs)
        else:
            raise Exception('Weighting {} not supported.'.format(self.weighting))

    def _tokenize(self, document):
        return document.split()

    # TODO: refeactor to get_document_token_matrix
    def get_document_token_matrix(self, documents):
        """
        Return the weighted document-token matrix for the given documents.

        :param documents: A list of documents that adhears to the same restricitions as the training documents, i.e. already preprocessed, tokenized, and splittable.
        :type documents: list
        :return: weighted document-token-matrix of shape num_documents x num_tokens
        :rtype: np.ndarray
        """
        document_token_matrix = self.sk_model.transform(documents)
        return document_token_matrix

    def get_id2token_mapping(self):
        """
        Return the mapping of ids to tokens. The id is the index of the column that represents a term in the document-term matrix.

        :return: Mapping of the unique token id to the unique token string. The inverse of the tok2id_mapping attribute.
        :rtype: dict(int: string)
        """
        return {v: k for k, v in self.token2id_mapping.items()}

    def get_token_list(self):
        """
        Return a list of tokens in order of the columns of the document-token matrix.

        :return: List of unique token strings
        :rtype: list(string)
        """
        id2token = self.get_id2token_mapping()
        token_list = [id2token[key] for key in sorted(id2token)]
        return token_list

    def save(self, path):
        #TODO: Describe storage documentation together with Topic Models in one central page of the documentation.
        """
        Save the model to disk.

        :param path: Relative path to the model. The path is appended to 'models/vectorizer/'
        :type path: string
        """

        full_path = os.path.join(ROOT_DIR, 'models/vectorizer', path)
        with open(full_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        Load a stored Vectorizer instance.

        :param path: Relative path to the stored model. The path is appended to 'models/vectorizer/'
        :type path: string
        :return: Vectorizer object
        """
        full_path = os.path.join(ROOT_DIR, 'models/vectorizer', path)
        with open(full_path, 'rb') as f:
            vec = pickle.load(f)
        return vec


if __name__ == '__main__':
    docs = ['I only by organic', 'organic food is a scam', 'organic food vs. conventional food']
    vec = Vectorizer('tfidf', docs)
    dtm = vec.get_document_token_matrix(docs)
    for i, val in enumerate(dtm):
        print("DocId {}".format(i), val.toarray())
