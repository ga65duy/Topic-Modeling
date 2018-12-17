import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pyLDAvis
import pandas as pd
import operator
import random
"""
Functions to analyze and visualize topic models
"""


def get_top_topics_for_documents(document_topic_matrix, num_top_topics=None, threshold=None, documents=None,
                                 values=False):
    """
    Return the top N topics or all topics above a certain treshold for every document.

    :param document_topic_matrix:
    :param num_top_topics: The top number of topics to retrieve for each document. Either this or :param threshold: has to be provided.
    :param threshold: Probability threshold. Topics that have a probability greater or equal to the threshold will be included. Note that cutting of by a threshold means that the documents have different numbers of topics assigned. Either this or :param num_top_topics: has to be provided. If both arguments are passed the latter will be used.

    :param documents: Ids of the documents to retrieve top topics for. A document id should correspond to the row of the document in the document-topic matrix.
    :param values: If true (topic_id, probability) tuples are returned. The probability is rounded to 4 digits.
    :return: list of lists
    """
    num_documents, num_topics = document_topic_matrix.shape

    if num_top_topics is None and threshold is None:
        raise Exception('Num_top_topics or threshold has to be passed.')
    elif num_top_topics is None and threshold is not None:
        threshold_passed = True
        num_top_topics = num_topics

    # Only num_top_topics or both num_top_topics and threshold are passed
    else:
        threshold_passed = False

    if not documents:
        documents = list(range(0, num_documents))

    if num_top_topics > num_topics:
        num_top_topics = num_topics

    top_topics_list = []
    documents = sorted(documents)

    for doc_id in documents:
        document_row = document_topic_matrix[doc_id, :]

        topics_ids_sorted = np.argsort(document_row)[::-1]
        if threshold_passed:
            probabilites_sorted = document_row[topics_ids_sorted]
            topics_ids_sorted = topics_ids_sorted[np.where(probabilites_sorted >= threshold)]
        else:
            topics_ids_sorted = topics_ids_sorted[:num_top_topics]

        if values:
            topic_list = [(idx, round(document_row[idx], 4)) for idx in topics_ids_sorted]
        else:
            topic_list = topics_ids_sorted.tolist()

        top_topics_list.append(topic_list)
    return top_topics_list


def get_document_count_for_topics(document_topic_matrix, num_top_topics=None, threshold=None):
    """
    For every topic return the number of documents that contain this topic with a certain probability threshold or among
    the top n topics of the documents.

    :param document_topic_matrix: The inferred document_topic_matrix to analyze
    :param num_top_topics: The number of top topics for every document to count the occurences of topics. Either this
    :param threshold:
    :return:
    """
    num_documents, num_topics = document_topic_matrix.shape

    if num_top_topics is None and threshold is None:
        raise Exception('Num_top_topics or threshold has to be passed.')
    elif num_top_topics is None and threshold is not None:
        top_topics_for_documents = get_top_topics_for_documents(document_topic_matrix, threshold=threshold)

    # Only num_top_topics or both num_top_topics and threshold are passed
    else:
        top_topics_for_documents = get_top_topics_for_documents(document_topic_matrix, num_top_topics=num_top_topics)

    topic_document_counts = []
    for topic in range(0, num_topics):
        doc_count = 0
        for doc in top_topics_for_documents:
            if topic in doc:
                doc_count += 1
        topic_document_counts.append(doc_count)
    return topic_document_counts


def get_documents_for_topics_with_treshold(document_topic_matrix, threshold=None, count=False, values=False):
    """
    Return the document ids for each topic where the topic occurs with a probability greater or equal to the threshold.
    If count is True the number of documents for each topic is returned.

    :param document_topic_matrix:
    :param threshold:
    :return:
    """
    topics = []
    for topic_id, documents in enumerate(document_topic_matrix.T):
        doc_ids = []
        for doc_id, topic_prob in enumerate(documents):
            if topic_prob >= threshold:
                doc_ids.append((doc_id, topic_prob))

        doc_ids.sort(key=operator.itemgetter(1), reverse=True)
        if not values:
            doc_ids = [i for i, prob in doc_ids]

        if count:
            topics.append(len(doc_ids))
        else:
            topics.append(doc_ids)
    return topics


def _compute_linear_trend_per_topic(doc_counts_per_year_per_topic):
    """
    Compute the trend of a topic by fitting a line to the document counts for every year.
    :param doc_counts_per_year_per_topic:
    :return:
    """
    gradient_per_topic = []
    for topic in doc_counts_per_year_per_topic:
        x = np.arange(0, len(topic))
        gradient = np.polyfit(x, topic, 1)[0]
        gradient_per_topic.append(gradient)

    gradient_per_topic = np.asarray(gradient_per_topic)
    # sort topic ids ascending by gradient
    topic_ids_sorted = gradient_per_topic.argsort()
    return topic_ids_sorted


def _get_documents_per_year_for_topics(document_topic_matrix, year2document, threshold=None):
    """
    For every topic and every year return the document_ids that contain the topic greater or equal to the threshold split by the years.

    :param document_topic_matrix:
    :param year2document:
    :param threshold:
    :return:
    """
    topics_documents = get_documents_for_topics_with_treshold(document_topic_matrix, threshold)
    topic_year_docids = []
    for topic in topics_documents:
        year_docs = [set(topic) & set(year2document[year]) for year in year2document]
        topic_year_docids.append(year_docs)
    return topic_year_docids


def _get_document_counts_per_year_for_topics(document_topic_matrix, year2document, normalize=False,
                                            total_articles_per_year=None, threshold=None):
    """

    :param document_topic_matrix:
    :param year2document:
    :param normalize:
    :param total_articles_per_year:
    :param threshold:
    :return:
    """
    topic_year_docids = _get_documents_per_year_for_topics(document_topic_matrix, year2document, threshold)
    topics_year_counts = []
    for topic in topic_year_docids:
        topic_year_counts = [len(year) for year in topic]
        topics_year_counts.append(topic_year_counts)

    years = range(2007, 2018)
    if normalize and total_articles_per_year:
        total_articles_per_year = [val for key, val in sorted(total_articles_per_year.items())]
        # print(total_articles_per_year)
        assert(len(total_articles_per_year) == len(years))
        topics_year_counts = np.asarray(topics_year_counts) / np.asarray(total_articles_per_year)
        topics_year_counts = topics_year_counts.tolist()
    elif normalize and not total_articles_per_year:
        raise Exception('If normalized is True, the total_articles_per_year have to be passed')

    return topics_year_counts


def plot_topic_time_distribution(document_topic_matrix, year2document, threshold=0.3, num_topics=None, trend=None, topic_ids=None, normalize=False, total_articles_per_year=None, path=None , labels=None):
    """
    Plot the distribution of topics over time. For detailed information see thesis Chapter 6.

    :param document_topic_matrix:
    :param year2document: Mapping of year to document ids. Can be created by calling map_document_to_year() on data_loader.
    :param threshold: The minimum probability a document should express the topics to be counted.
    :param num_topics: The number of topics to plot.
    :param trend: 'Rising' or 'Declining'
    :param normalize: If true the document counts per topic and year will be divided by the total document counts in the year. If true, total_articles_per_year have to passed.
    :param total_articles_per_year: The total articles per year. Can be created by calling map_document_to_year(counts=True) on data_loader
    :param path: Path to store the resulting figure. It should include a valid matplotlib file ending.
    :return:
    """

    topic_years_doc_counts = _get_document_counts_per_year_for_topics(document_topic_matrix, year2document, normalize, total_articles_per_year, threshold=threshold)
    topic_trends = _compute_linear_trend_per_topic(topic_years_doc_counts)
    years = range(2007, 2018)

    if trend and not topic_ids:
        if trend == 'rising':
            topic_ids = topic_trends[-num_topics:]
        elif trend == 'declining':
            topic_ids = topic_trends[:num_topics]
        elif trend == 'stable':
            median_ind = len(topic_trends) // 2
            of = num_topics // 2
            topic_ids = topic_trends[median_ind-of: median_ind+of]
        else:
            raise Exception('Trend {} not supported'.format(trend))
    elif trend and topic_ids:
        raise Exception('You can only pass a trend or topics')
    elif not trend and not topic_ids:
        raise Exception('You have to pass either trend or topic')

    for ind, topicid in enumerate(topic_ids):
        if labels is not None:
            label = 'Topic #{} - {}'.format(topicid, labels[ind])
        else:
            label = 'Topic #{}'.format(topicid)
        plt.plot(years, topic_years_doc_counts[topicid], label=label)
    plt.xticks(years)
    if trend:
        plt.title("Top {} {} topics".format(num_topics, trend).title())
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)

    if path:
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()


def get_document_topic_dataframe(document_topic_matrix, titles, sources):
    """
    Convert a document_topic_matrix to a Pandas DataFrame including the source and title for each document. The dataframe
    can be used to match the results of topic modeling to the results of other tasks.

    :param document_topic_matrix: A document-topic matrix as returned by calling get_document_topic_matrix() on a topic model.
    :type document_topic_matrix: np.ndarray
    :param titles: A list of titles. The length of the title list must equal the number of documents in the document_topic_matrix.
    The index of a title in the list must equal the row of the corresponding document in the document_topic_matrix.
    :type titles: list(str)
    :param source: A list of sources. The length of the source list must equal  the number of documents in the document_topic_matrix.
    The index of a source in the list must equal to the row of the corresponding document in the document_topic_matrix.
    :type source: list(str)
    :return: A DataFrame with the columns [Source, Title, Topic 0, ..., Topic N]
    :rtype: pd.DataFrame
    """
    num_docs, num_topics = document_topic_matrix.shape
    column_labels = ['Topic {}'.format(i) for i in range(0, num_topics)]
    df = pd.DataFrame(document_topic_matrix, columns=column_labels)
    df.insert(len(df.columns), 'Title', titles)
    df = df.reindex(columns=(['Title'] + list([a for a in df.columns if a != 'Title'])))
    df.insert(len(df.columns), 'Source', sources)
    df = df.reindex(columns=(['Source'] + list([a for a in df.columns if a != 'Source'])))
    return df


def print_topic_documents(topic_model, document_topic_matrix, num_top_tokens=10, titles=None, dates=None, threshold=0.5):
    """
    Print the num_top_tokens for each topic, and the titles and dates of the documents that contain the topic with a probability
    greater or equal to the threshold. A csv is printed to the output that can be directly imported to Excel.

    For the qualitative analysis of the topic models the output of this visualization together with the top 10 words
     for each topic were presented to domain experts.

    :param topic_model: A trained topic_model.
    :param document_topic_matrix: A document-topic matrix as returned by calling get_document_topic_matrix() on a topic model.
    :type document_topic_matrix: np.ndarray
   :param num_top_tokens: Number of top words for each topic.
    :type num_top_tokens: int
    :param titles: A list of titles. The length of the title list must equal the number of documents in the document_topic_matrix.
    The index of a title in the list must equal the row of the corresponding document in the document_topic_matrix.
     :type titles: list(str)
    :param dates: A list of dates. The length of the date list must equal the number of documents in the document_topic_matrix.
    The index of a date in the list must equal the row of the corresponding document in the document_topic_matrix.
    :type dates: list(str)
    :param threshold: The probability that a topic has to have in a document so that the document is listed as relevant for the topic.
    If no value is passed the default of 0.5 is assumed.
    :type threshold: int
    """
    num_documents, num_topics = document_topic_matrix.shape
    assert(num_documents == len(titles))
    assert(num_documents == len(dates))

    topics = topic_model.get_topics(num_top_tokens)
    documents_for_topic = get_documents_for_topics_with_treshold(document_topic_matrix, threshold, values=True)
    assert(len(topics) == len(documents_for_topic))
    for topic_id in range(len(topics)):
        print('Topic {} \t'.format(topic_id) + '\t'.join(topics[topic_id]))
        document_ids = documents_for_topic[topic_id]
        for doc_id, topic_probs in document_ids:
            print('DocId {}'.format(doc_id) + '\t' + str(topic_probs) + '\t\"' + titles[doc_id] + "\"\t" + dates[doc_id])
        print()


def plot_document_topic_histogram(document_topic_matrix, document_id, title=None, file=None):
    """
    Plot a histogram of the topic distribution for a document.
     :param document_topic_matrix: A document-topic matrix as returned by calling get_document_topic_matrix() on a topic model.
    :type document_topic_matrix: np.ndarray
    :param document_id: Id of the document to visualize. The id of a document  corresponds to the row in the document_topic_matrix i.e. the position in the list of documents.
    :type document_id: int
    :param title: Optional title of the document. If no title is passed the title of the plot is "Document <document_id>"
    :type title: str
    :param file: If a string is passed the plot is stored at PROJECT_ROOT/figures/histograms/<file>. If no file is passed the plot is shown.
    """
    num_docs, num_topics = document_topic_matrix.shape
    doc = document_topic_matrix[document_id,]
    top = np.arange(num_topics)
    fig, ax = plt.subplots()
    ax.bar(top, doc)
    x_labels = ['Topic {}'.format(t) for t in top]
    if title:
        ax.set_title(title.title())
    else:
        ax.set_title('Document {}'.format(document_id))
    ax.set_ybound(0, 1)
    ax.grid(axis='y')
    ax.set_xticks(top)
    ax.set_xticklabels(x_labels, rotation=270)
    for label in ax.get_xticklabels()[1::2]:
        label.set_visible(False)
    plt.tight_layout()
    #if file:
        #path = os.path.join(REPORT_DIR, 'figures/histograms', file)
        #plt.savefig(path)
    #else:
    plt.show()


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



def get_word_intrusion_dataframe(topic_model, num_top_tokens=5):
    """
    Return a dataframe with six words per topic. Five of the words are the words with the highest probability for the topic while the sixth word  is random word that has a high probability (top 5) in another topic. This resembles the word intrusion task by Chang et. al. (2009) to quantify the coherence of topics.

    :param topic_model: A trained topic_model that should be evaluated
    :param num_top_tokens: Number of top words for each topic. The resulting dataframe will always contain one additional term.
    :type num_top_tokens: int
    :return: pd.DataFrame

    """
    num_topics = topic_model.num_topics
    topics_df = topic_model.get_topics_dataframe(num_top_tokens)
    intruders = []

    for topic_id in range(num_topics):
        other_topic_ids = [i for i in range(num_topics) if i != topic_id]

        selected_id = random.choice(other_topic_ids)
        other_topic = topics_df.iloc[selected_id, :]
        intruder = other_topic.sample(n=1).iloc[0]
        intruders.append(intruder)
    assert(len(intruders) == num_topics)
    intruders_df = topics_df.assign(intruder=intruders)
    intruders_df2= intruders_df.copy()

    intruders_df.columns = [i for i in range(num_top_tokens + 1)]
    intruders_list = intruders_df.values.tolist()
    intruders_shuffled = []
    for row in intruders_list:
        random.shuffle(row)
        intruders_shuffled.append(row)
    row_labels = ['Topic {}'.format(i) for i in range(len(intruders_shuffled))]
    permutated_df = pd.DataFrame(intruders_shuffled, index=row_labels)

    return intruders_df2, permutated_df

def plot_training_convergence(topic_models, method, dataset, log=False, path=None):
    """
    Plot the convergence values over epochs for a list of topic models.
    This should be called with topic models of the same type and trained on the same data but with different
    number of topics.

    :param topic_models: List of trained topic models.
    :param method: Type of topic modeling algorithm. "nmf" or "lda"
    :type method: str
    :param dataset: The type of data this model was trained with. Used as the title of the plot.
    :type dataset: str
    :param path: If a path is passed the plot will be stored in reports/figures/training/<method>/<path>. Make sure that the path contains
    a valid file extension for matplotlib plots.
    :type path: str

    """
    for model in topic_models:
        convergence = model.get_training_convergence()
        num_epochs = range(len(convergence))
        num_topics = model.num_topics
        plt.plot(num_epochs, convergence, '.-', label='{} Topics'.format(num_topics))

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid(True, linestyle='dotted')
    plt.xlabel('Epochs')
    plt.xticks(num_epochs)
    if method.lower() == 'nmf':
        plt.ylabel('Error')
    elif method.lower() == 'lda':
        plt.ylabel('Perplexity')
    else:
        raise Exception('{} not supported'.format(method.lower()))

    if log:
        plt.yscale('log')

    plt.title(dataset)
    plt.show()

def plot_coherence_values(topic_models, range_topics, dataset, path=None):
    """
    Plot the coherence values over the number of topics on the same dataset.
     This should be called with topic models of the same type and trained on the same data but with different
    number of topics.

    :param topic_models: List of trained topic models.
    :param dataset: The type of data this model was trained with. Used as the title of the plot.
    :type dataset: str
    :param path: If a path is passed the plot will be stored in reports/figures/training/<path>. Make sure that the path contains
    a valid file extension for matplotlib plots.
    :type path: str
    """
    assert(len(topic_models) == len(range_topics))
    coherence_values = []
    for tm in topic_models:
        coherence_values.append(tm.get_topic_coherence())
    assert(len(coherence_values) == len(range_topics))
    plt.plot(range_topics, coherence_values, '.-')

    plt.grid(True, linestyle='dotted')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence')
    plt.title(dataset)

    #if path:
        #plt.savefig(os.path.join(REPORT_DIR, 'figures/training', path))
    #else:
    plt.show()

def plot_lda_nmf_coherence_values(lda_models, nmf_models, range_topics, dataset, path=None):
    """
    Plot the coherence values per number of topics.
    :param lda_models: List of trained LDA models
    :param nmf_models: List orained NMF models
    :param range_topics: List of the number of topics. The list length of range_topics, lda_models, and nmf_models must be the same.
    :param dataset: Dataset (e.g. 'English editorial articles') the models were trained with.
    :param path: If a path is passed the plot will be stored in reports/figures/training/<path>. Make sure that the path contains
    a valid file extension for matplotlib plots.
    """
    lda_coherence_values = []
    for tm in lda_models:
        lda_coherence_values.append(tm.get_topic_coherence())
    assert (len(lda_coherence_values) == len(range_topics))

    nmf_coherence_values = []
    for tm in nmf_models:
        nmf_coherence_values.append(tm.get_topic_coherence())
    assert (len(nmf_coherence_values) == len(range_topics))

    plt.plot(range_topics, lda_coherence_values, '.-', label='LDA')
    plt.plot(range_topics, nmf_coherence_values, '.-', label='NMF')

    plt.legend(loc='best')
    plt.grid(True, linestyle='dotted')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence')
    plt.title(dataset)

    #if path:
        #plt.savefig(os.path.join(REPORT_DIR, 'figures/training', path), bbox_inches='tight')
    #else:
    plt.show()


def plot_topic_max_probability_distribution(document_topic_matrix):
    """
    For every document extract the value of the most probable topic and plot the counts for every value in a histogram.
    :param document_topic_matrix
    """
    num_docs, num_topics = document_topic_matrix.shape
    max_probs = np.amax(document_topic_matrix, axis=1)
    mean = float(np.round(np.mean(max_probs), 3))
    var = float(np.round(np.var(max_probs), 3))
    sns.distplot(max_probs, bins=np.linspace(0, 1, 20), kde=False).set_title('{} topics, mean={}, var={}'.format(num_topics, mean, var))
    plt.xlabel('Values for Most Probable Topic')
    plt.ylabel('Number of Documents')
    plt.show()
    return mean


def plot_document_counts_per_run(topic_models, doc_topic_matrices, range_topics, threshold, title, path):
    """
    See thesis Chapter 5 on model selection by the number of topics. Checkout notebooks/evaluation for example usage.

    """
    assert(len(doc_topic_matrices) == len(range_topics))
    doc_sums = []
    for idx, document_topic_matrix in enumerate(doc_topic_matrices):
        num_docs_per_topic = get_document_count_for_topics(document_topic_matrix, threshold=threshold)
        topics_ranked = topic_models[idx].get_topics_ranked(num_docs_per_topic).iloc[0:50, ]
        doc_sum = topics_ranked['score'].sum()
        doc_sums.append(doc_sum)

    sns.set()
    x_ticks = [i for i in range(len(range_topics))]
    plt.title(title)
    plt.xlabel('Number of Topics')
    plt.ylabel('Number of Documents')
    plt.bar(x_ticks, doc_sums, color='gray')
    plt.xticks(x_ticks, range_topics)
    plt.show()
