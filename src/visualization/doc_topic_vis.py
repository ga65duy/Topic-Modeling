import src.data.data_loader as dl
from src.features.vectorizer import Vectorizer
from src.models.topic_models import TopicModel
from collections import defaultdict
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

model_dict = {"topic_models/lda/DEED_lda_german_editorial_articles_190.pkl": ["lda", "german", "editorial"],
              "topic_models/lda/ENED_lda_english_editorial_articles_130.pkl": ["lda", "english", "editorial"],
              "topic_models/lda/ENFO_lda_english_forum_110.pkl": ["lda", "english", "forum"],

              "topic_models/nmf/DEEDCO_nmf_german_editorial_comments_170.pkl": ["nmf", "german", "editorial"],
              "topic_models/nmf/DEFO_nmf_german_forum_170.pkl": ["nmf", "german", "forum"],
              "topic_models/nmf/ENEDCO_nmf_english_editorial_comments_170.pkl": ["nmf", "english", "editorial"]}


def get_document_topic_matrix(model_pkl):
    """
    Load a Topic model and get the document topic matrix

    :param model_pkl: string to load a model
    :return: the document topic matrix and the loaded model
    """
    if model_pkl == "topic_models/lda/DEED_lda_german_editorial_articles_190.pkl":
        data = dl.get_articles_by_type(model_dict[model_pkl][1], model_dict[model_pkl][2])
        texts = data['article_texts']
        vec = Vectorizer.load("vectorizer/DEED_lda_german_editorial.pkl")
        model = TopicModel.load(model_pkl)

    elif model_pkl == "topic_models/lda/ENED_lda_english_editorial_articles_130.pkl":
        data = dl.get_articles_by_type(model_dict[model_pkl][1], model_dict[model_pkl][2])
        texts = data['article_texts']
        vec = Vectorizer.load("vectorizer/ENED_lda_english_editorial.pkl")
        model = TopicModel.load(model_pkl)

    elif model_pkl == "topic_models/lda/ENFO_lda_english_forum_110.pkl":
        data = dl.get_forum_threads_by_language(model_dict[model_pkl][1])
        texts = data['thread_texts']
        vec = Vectorizer.load("vectorizer/ENFO_lda_english_forum.pkl")
        model = TopicModel.load(model_pkl)


    elif model_pkl == "topic_models/nmf/DEEDCO_nmf_german_editorial_comments_170.pkl":
        data = dl.get_comments_by_type(model_dict[model_pkl][1], model_dict[model_pkl][2])
        texts = data['comment_texts']
        vec = Vectorizer.load("vectorizer/DEEDCO_nmf_german_editorial.pkl")
        model = TopicModel.load(model_pkl)

    elif model_pkl == "topic_models/nmf/DEFO_nmf_german_forum_170.pkl":
        data = dl.get_forum_threads_by_language(model_dict[model_pkl][1])
        texts = data['thread_texts']
        vec = Vectorizer.load("vectorizer/DEFO_nmf_german_forum.pkl")
        model = TopicModel.load(model_pkl)

    elif model_pkl == "topic_models/nmf/ENEDCO_nmf_english_editorial_comments_170.pkl":
        data = dl.get_comments_by_type(model_dict[model_pkl][1], model_dict[model_pkl][2])
        texts = data['comment_texts']
        vec = Vectorizer.load("vectorizer/ENEDCO_nmf_english_editorial.pkl")
        model = TopicModel.load(model_pkl)

    else:
        raise Exception('Models for {} not found'.format(model_pkl))

    document_term_matrix = vec.get_document_token_matrix(texts)

    return model.get_document_topic_matrix(document_term_matrix), model



def amount_topic_per_dokument(min_probability, matrix):
    """"
    Calculate the amount of topics in one document

    :param min_probability: a min percentage of a topic in a document
    :type: int
    :param: matrix: the document topic matrix
    :type: matrix
    :return:list with the the amount of topics in a document
    """
    newMatrix = []
    for matrixrow in range(0, len(matrix)):
        count = 0
        for matrixcolumn in range(0, (len(matrix[0]))):
            if matrix[matrixrow, matrixcolumn] >= min_probability:
                count += 1
        newMatrix.append(count)

    amount_of_topics, number_of_documents = count_amount_of_topics(newMatrix)
    return amount_of_topics, number_of_documents


def count_amount_of_topics(amount_topic_per_dokument):
    """"

    :param: amount_topic_per_document
    :type matrix
    :return 2 lists. The first list is the amount of topics
                    the second list is the number of documents
    """

    doc_count_dict = defaultdict(int)
    for doc in amount_topic_per_dokument:
        doc_count_dict[doc] += 1

    asc = OrderedDict(sorted(doc_count_dict.items()))
    k = [k for (k, v) in asc.items()]
    v = [v for (k, v) in asc.items()]
    return k, v


def plot_amount_topic_per_document(model_pkl, threshold, topic_amount, document_amount):
    """
    Plots how many topics, which are over the threshold, occure in how many documents.

    :param model_pkl: string to load the Topic Model
    :param threshold: minimal percentage a topic has to occure
    :type float
    :param topic_amount:  how many topics are in certain document
    :type list
    :param document_amount: sumed up documents which have a certain amount of topics
    :type list
    :return: Plot
    """
    plt.bar(topic_amount, document_amount, 0.9)
    plt.title('Number of documents with amount of topics \n with threshold of {}% for {} {} with {} model'.format(
        threshold * 100, model_dict[model_pkl][1], model_dict[model_pkl][2], model_dict[model_pkl][0]))
    plt.xlabel('Amount of Topics')
    plt.ylabel('Number of documents')
    # ohne xtickts wird die breite automatisch generiert(nicht einheitlich)
    plt.xticks(topic_amount, topic_amount)
    plt.show()


def amount_doc_per_topic(min_probability, matrix):
    """

    :param min_probability: minimal percentage a topic has to coocure in a document
    :type float
    :param matrix: document term matrix
    :type matrix
    :return: a list in how many documents the topic occurs
    """
    newMatrix = []
    for matrixcolumn in range(0, (len(matrix[0]))):
        count = 0
        for matrixrow in range(0, len(matrix)):
            if matrix[matrixrow, matrixcolumn] >= min_probability:
                count += 1
        newMatrix.append(count)
    return newMatrix


def amount_doc_per_topic_sorted(min_probability, matrix):
    """

    :param min_probability:minimal percentage a topic has to coocure in a document
    :param matrix:
    :return:
    """
    newMatrix = []
    topic_index = 0
    for matrixcolumn in range(0, (len(matrix[0]))):
        count = 0
        for matrixrow in range(0, len(matrix)):
            if matrix[matrixrow, matrixcolumn] >= min_probability:
                count += 1
        newMatrix.append((topic_index, count))
        topic_index += 1
    # sortierung nach dem 2ten tupl
    sorted_by_second = sorted(newMatrix, key=lambda tup: tup[1], reverse=True)

    tid_list = [k for (k, v) in sorted_by_second]
    doc_count_list = [v for (k, v) in sorted_by_second]

    return tid_list, doc_count_list


def plot_amount_doc_per_topic(model_pkl, threshold, limit_x,
                              sorted=True):
    """
    Plot how often a topic occurs in different documents
    :param model_pkl: string to load a topic Model
    :param topicmodel:
    :param threshold: minimal percentage a topic has to coocure in a document
    :param document_topic_matrix:
    :param limit_x: number of values shown on the x-axis
    :param sorted: Sorts descending
    :return:
    """
    plt.title('Number of documents with amount of topics \n with threshold of {}% for {} {} with {} model'.format(
        threshold * 100, model_dict[model_pkl][1], model_dict[model_pkl][2], model_dict[model_pkl][0]))
    plt.xlabel('Topicnumber')
    plt.ylabel('Number of documents')

    document_topic_matrix, topicmodel = get_document_topic_matrix(model_pkl)
    if sorted:
        tid_list, doc_count_list = amount_doc_per_topic_sorted(threshold, document_topic_matrix)
        topicnumber = range(0, len(tid_list))
        plt.bar(topicnumber, doc_count_list, 0.9)

        if limit_x < 50:
            plt.xticks(tid_list, tid_list, rotation=45)
        else:
            plt.xticks([], [])

        # plt.xticks(range(0,len(tid_list)), tid_list, rotation=45)
        plt.xlim(-1, limit_x)
        plt.show()

        # plottable
        table = topicmodel.get_topics_dataframe()
        display(table.iloc[tid_list[:limit_x + 1], :])

    else:
        document_topic_list = amount_doc_per_topic(threshold, document_topic_matrix)
        topicnumber = range(0, len(document_topic_list))
        plt.bar(topicnumber, document_topic_list, 0.9)

        # plt.xticks(np.arange(0, len(document_topic_list), step=50))
        if limit_x < 50:
            plt.xticks(topicnumber, document_topic_list)
        else:
            plt.xticks([], [])

    plt.show()


def plot_sumed_theta(model_pkl, sorted=True):
    """
    Plot with the summed up theta values from the document topic matrix

    :param sorted: True if the summed up thetavalues shall be sorted descending else false
    :type boolean
    :param model: string lda or nmf
    :param language: string german or english
    :param typetx: string editorial or forum
    :return plot
    """
    tm,summed_up_columns = get_summed_theta(model_pkl)

    plt.title('Summed up theta per topic for {}'.format(model_dict[model_pkl][0]))
    plt.xlabel('Topic')
    plt.ylabel('Theta')

    if sorted:
        indices = np.argsort(summed_up_columns)[::-1]
        sorted_t = summed_up_columns[indices]
        plt.xticks([], [])
        plt.bar(range(0, len(sorted_t)), sorted_t, 0.3)
    else:
        plt.bar(range(0, len(summed_up_columns)), summed_up_columns, 0.3)
        # plt.xticks(range(0, len(entropy)),range(0, len(entropy)), rotation=45)
    plt.show()
    # plt.savefig("test.png",bbox_inches = "tight")


def get_summed_theta(model_pkl):
    """

    :param model_pkl: string to load a topic model
    :return:
    """
    document_topic_matrix, tm = get_document_topic_matrix(model_pkl)
    summed_up_columns = np.around(np.sum(document_topic_matrix, axis=0),decimals = 3)
    return tm,summed_up_columns


if __name__ == "__main__":
    model_pkl = "topic_models/lda/ENED_lda_english_editorial_articles_130.pkl"
    threshold = 0.1
    x_limit = 20

    document_topic_matrix, topicmodel = get_document_topic_matrix(model_pkl)
    k, v = amount_topic_per_dokument(threshold, document_topic_matrix)
    plot_amount_topic_per_document(model_pkl, threshold, k, v)
    plot_amount_doc_per_topic(model_pkl, threshold, 190,
                              sorted=True)
    plot_amount_doc_per_topic(model_pkl, threshold, x_limit,
                              sorted=True)
