import src.data.data_loader as dl
from src.features.vectorizer import Vectorizer
from src.models.topic_models import TopicModel
from collections import defaultdict
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display


def get_document_topic_matrix(model, language, typetx):
    """"
    return the document topic matrix for the choosen model, language and typetx
    
    :param model: string lda or nmf
    :param language: string german or english
    :param typetx: string editorial or forum
    """
    data = dl.get_articles_by_type(language, typetx)
    texts = data['article_texts']

    if model == "lda":
        if language == "german" and typetx == "editorial":
            lda_vec = Vectorizer.load("vectorizer/DEED_lda_german_editorial.pkl")
            lda = TopicModel.load("topic_models/lda/DEED_lda_german_editorial_articles_190.pkl")

        elif language == "english" and typetx == "editorial":
            lda_vec = Vectorizer.load("vectorizer/ENED_lda_english_editorial.pkl")
            lda = TopicModel.load("topic_models/lda/ENED_lda_english_editorial_articles_130.pkl")

        elif language == "english" and typetx == "forum":
            lda_vec = Vectorizer.load("vectorizer/ENFO_lda_english_forum.pkl")
            lda = TopicModel.load("topic_models/lda/ENFO_lda_english_forum_110.pkl")

        lda_document_term_matrix = lda_vec.get_document_token_matrix(texts)
        return lda.get_document_topic_matrix(lda_document_term_matrix), lda

    elif model == "nmf":
        if language == "german" and typetx == "editorial":
            nmf_vec = Vectorizer.load("vectorizer/DEEDCO_nmf_german_editorial.pkl")
            nmf = TopicModel.load("topic_models/nmf/DEEDCO_nmf_german_editorial_comments_170.pkl")

        elif language == "german" and typetx == "forum":
            nmf_vec = Vectorizer.load("vectorizer/DEFO_nmf_german_forum.pkl")
            nmf = TopicModel.load("topic_models/nmf/DEFO_nmf_german_forum_170.pkl")

        elif language == "english" and typetx == "editorial":
            nmf_vec = Vectorizer.load("vectorizer/ENEDCO_nmf_english_editorial.pkl")
            nmf = TopicModel.load("topic_models/nmf/ENEDCO_nmf_english_editorial_comments_170.pkl")

        nmf_document_term_matrix = nmf_vec.get_document_token_matrix(texts)
        return nmf.get_document_topic_matrix(nmf_document_term_matrix), nmf
    else:
        raise Exception('Source {} not included'.format(typetx))


# limitation per row
# wie viele topics die grenze pro doc überschritten
def amount_topic_per_dokument(min_probability, matrix):
    """"
    returns a list with the the amount of topics in a document
    :param min_probability: a min percentage of a topic in a document
    :typetx: int
    :param: matrix: the document topic matrix
    :typetx: list
    """
    newMatrix = []
    for matrixrow in range(0, len(matrix)):
        count = 0
        for matrixcolumn in range(0, (len(matrix[0]))):
            if matrix[matrixrow, matrixcolumn] >= min_probability:
                count += 1
        newMatrix.append(count)
    max_topic_number = max(newMatrix)

    # return newMatrix, max_topic_number

    amount_of_topics, number_of_documents = count_amount_of_topics(newMatrix)
    return amount_of_topics, number_of_documents


def count_amount_of_topics(amount_topic_per_dokument):
    """"
    returns 2 lists. the first list is the amount of topics 
                    the second list is the number of documents

    param: amount_topic_per_document
    typetx: list
    """

    doc_count_dict = defaultdict(int)
    for doc in amount_topic_per_dokument:
        doc_count_dict[doc] += 1

    asc = OrderedDict(sorted(doc_count_dict.items()))
    k = [k for (k, v) in asc.items()]
    v = [v for (k, v) in asc.items()]
    return k, v


# model,language,typetx,
def plot_amount_topic_per_document(model,language,typetx,threshold, k, v):
    plt.bar(k, v , 0.9)
    plt.title('Number of documents with amount of topics \n with threshold of {}% for {} {} with {} model'.format(threshold*100, language, typetx, model))
    plt.xlabel('Amount of Topics')
    plt.ylabel('Number of documents')
    #ohne xtickts wird die breite automatisch generiert(nicht einheitlich)
    plt.xticks(k, k)
    plt.show()


# limitation per column
def amount_doc_per_topic(min_probability, matrix):
    """
    returns a list in how many documents the topic occurs
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


def plot_amount_doc_per_topic(model,language,typetx,topicmodel,threshold, document_topic_matrix, limit_x, sorted=True):

    plt.title('Number of documents with amount of topics \n with threshold of {}% for {} {} with {} model'.format(threshold * 100, language, typetx, model))
    plt.xlabel('Topicnumber')
    plt.ylabel('Number of documents')

    if sorted:
        tid_list, doc_count_list = amount_doc_per_topic_sorted(threshold, document_topic_matrix)
        topicnumber = range(0, len(tid_list))
        plt.bar(topicnumber, doc_count_list, 0.9)

        if limit_x < 50:
            plt.xticks(tid_list, tid_list,rotation = 45)
        else:
            plt.xticks([], [])

        #plt.xticks(range(0,len(tid_list)), tid_list, rotation=45)
        plt.xlim(-1, limit_x)
        plt.show()

        #plottable
        table = topicmodel.get_topics_dataframe()
        display(table.iloc[tid_list[:limit_x + 1], :])

    else:
        document_topic_list = amount_doc_per_topic(threshold, document_topic_matrix)
        topicnumber = range(0, len(document_topic_list))
        plt.bar(topicnumber, document_topic_list, 0.9)

        #plt.xticks(np.arange(0, len(document_topic_list), step=50))
        if limit_x <50:
            plt.xticks(topicnumber, document_topic_list)
        else:
            plt.xticks([], [])

    plt.show()



if __name__ == "__main__":
    model = "lda"
    language = "german"
    typetx = "editorial"
    threshold = 0.1
    x_limit = 20

    document_topic_matrix, topicmodel = get_document_topic_matrix(model, language, typetx)
    k, v = amount_topic_per_dokument(threshold, document_topic_matrix)
    plot_amount_topic_per_document(model, language, typetx, threshold, k, v)
    # vis.plot_amount_doc_per_topic(model,language,typetx,topicmodel,threshold, document_topic_matrix,x_limit, sorted = False)
    plot_amount_doc_per_topic(model, language, typetx, topicmodel, threshold, document_topic_matrix, 190,
                                  sorted=True)
    plot_amount_doc_per_topic(model, language, typetx, topicmodel, threshold, document_topic_matrix, x_limit,
                                  sorted=True)