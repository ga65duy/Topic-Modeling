import pyLDAvis
import os
import numpy as np
from src.models.topic_models import TopicModel

def plot_pyldavis(topic_model, document_topic_matrix, document_term_matrix, file=None, **kwargs):
    """
    Generate a pyLDAvis visualization of the given topic model. For more information about the visualization read the `original paper <http://www.aclweb.org/anthology/W14-3110>`_ by Sievert and Shirley. Note that pyLDAvis only supports LDA models,
     passing a nmf model will cause an exception.

    :param document_topic_matrix: A document-topic matrix as returned by calling get_document_topic_matrix() on a topic model.
    :type document_topic_matrix: np.ndarray
    :param document_term_matrix: Term count weighted document-term matrix of the documents used to infer the document_topic_matrix.
    :type document_term_matrix: np.ndarray
    :param file: Path to store the HTML output. If no file is passed the plot is visualized in the browser.
    :type file: str
    :param kwargs: Further parameters passed directly to pyLDAvis's prepare function. See the `documentation <http://pyldavis.readthedocs.io/en/latest/modules/API.html#pyLDAvis.prepare>`_ for options. Note, that sort_topics=False is already set.
    """
    if topic_model.model_name != 'lda':
        raise Exception('pyLDAvis only supports LDA. {} not supported'.format(topic_model.model_name))
    topic_token_matrix = topic_model.get_topic_token_matrix(normalize=True)
    id2word = topic_model.id2token

    document_lengths = np.sum(document_term_matrix, axis=1).getA1()
    term_frequencies = np.sum(document_term_matrix, axis=0).getA1()
    prepared_data = pyLDAvis.prepare(topic_token_matrix, document_topic_matrix, document_lengths, id2word,
                                     term_frequencies, sort_topics=False, **kwargs)

    ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    REPORT_DIR = os.path.join(ROOT_DIR, 'reports')
    if file:
        base_path = os.path.join(REPORT_DIR, 'figures/pyLDAvis')
        pa = os.path.join(base_path, file)
        with open(pa, 'w') as f:
         pyLDAvis.save_html(prepared_data, f)
    else:
        pyLDAvis.show(prepared_data)

if __name__ == "__main__":
    model = TopicModel.load('topic_models/lda/DEED_lda_german_editorial_articles_190.pkl')
    document_term_matrix =model.train_document_term_matrix
    document_topic_matrix = model.get_document_topic_matrix(document_term_matrix)

    print(plot_pyldavis(model, document_topic_matrix,document_term_matrix))