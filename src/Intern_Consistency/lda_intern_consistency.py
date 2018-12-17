from src.models.topic_models import TopicModel
import matplotlib.pyplot as plt
from src.Intern_Consistency.hLda_jenson_shannon import jensen_shannon
import numpy as np
import pandas as pd
from scipy.stats import entropy

"""
Calculate the Similatities for the same levels of a tree with Jenson Shannon for the Topic Models
"""

def get_alphas(model_pkl):
    """
    Returns the alphas from a given topic model

    :param model_pkl: string to load a model
    :return: the loaded topic model and the alphas per topic
    """

    if "lda" not in model_pkl:
        raise Exception("No alphas for model nmf")
    else:
        topicmodel = TopicModel.load(model_pkl)
        alphas = np.around(topicmodel.model.alpha,decimals = 3)
    return topicmodel,alphas

def plot_alphas_per_model(model_pkl,title,limit_x, sorted = True):
    """
    The alphas are plotted in a graph. It can be sorted according to the topic number or the alpha value.

    :param model_pkl: string to load a model
    :param sorted: if true the alphas are sorted descending
    type: boolean
    :return: a plot with the topics and their alphas
    """
    tm,alphas = get_alphas(model_pkl)

    plt.title("Alphas for {}".format(title))
    plt.xlabel('Topics')
    plt.ylabel('Alpha')

    if sorted:
        indices = np.argsort(alphas)[::-1]
        sorted_a = alphas[indices]
        if limit_x <= 50:
            plt.bar(range(0, len(sorted_a)), sorted_a, 0.3)
            plt.xticks(range(0,len(indices)), indices, rotation=90)
            plt.ylim(0, max(sorted_a) + 1)
            plt.xlim(-1, limit_x)
        else:
            plt.xticks([], [])
            plt.fill_between(range(0, len(indices)), sorted_a)
        plt.ylim(0, max(alphas) + 0.01)

    else:
        plt.bar(range(0, len(alphas)), alphas, 0.3)
    path =title.replace(" ","_")
    plt.savefig("D:\\Bachelorarbeit\\Thesis\\cleanthesis-TUM\\cleanthesis-TUM\\gfx\\Alphas\\{}_{}.pdf".format(path,limit_x),
                bbox_inches='tight')
    plt.show()
    #plt.savefig("test.png",bbox_inches = "tight")

def plot_sim_of_phi_matrix(model_pkl, sorted = True):
    """
    Shows the topics and their similarities (calculatet with 1- Jenson Shannon matrix)
    If sorted is true the similarities are sorted descending and a tuple with (topic, similarity) is in every row
    Plot the similarity (1-jensonshannon()) between all topics

    :param sorted: Sorts the similarities descending
    :type boolean
    :param model_pkl: string to load a model
    :return: a dataframe with the similarities
    """
    topicmodel = TopicModel.load(model_pkl)
    ttm = topicmodel.get_topic_token_matrix()

    matrix = np.zeros((len(ttm), len(ttm)))
    topic_num1 = []
    for i, tt in enumerate(ttm):
        topic_num1.append("Topic " + str(i))
        for j, t in enumerate(ttm):
            matrix[i, j] = 1 - jensen_shannon(tt, t)
    df = pd.DataFrame(matrix, columns=topic_num1, index=topic_num1)
    print("Compare {} with".format(model_pkl))
    if sorted:
        sort_arg = np.argsort(matrix, axis=0)[::-1]
        sorted_v = np.sort(matrix, axis=0)[::-1]
        result = []
        for c in range(0, len(matrix)):
            c_list = []
            for r in range(0, len(matrix)):
                t = "Topic " + str(sort_arg[r, c])
                v = round(sorted_v[r, c], 3)
                c_list.append((t, v))
            result.append(c_list)
        topic_label = ["Topic " + str(i) for i in range(0, len(matrix))]
        df2 = pd.DataFrame(result, index=topic_label)
        return df2
    else:
        return df

def plot_sim_of_phi2_matrix(model_pkl,model_pkl2, sorted = True, axis = "column"):
    """
    Shows the topics of tow different topic models and there similarities (calculatet with 1- Jenson Shannon matrix)
    If sorted is true the similarities are sorted descending and a tuple with (topic, similarity) is in every row
    Plots the similiarites of 2 topic models

    :param model_pkl: string to load a model
    :param model_pkl2: string to load a second model
    :return: a datafrmae with the similarities
    """
    topicmodel = TopicModel.load(model_pkl)
    topicmodel2 = TopicModel.load(model_pkl2)
    ttm = topicmodel.get_topic_token_matrix()
    ttm2 = topicmodel2.get_topic_token_matrix()
    matrix = np.zeros((len(ttm), len(ttm2)))
    topic_num1 = []
    for i, tt in enumerate(ttm):
        topic_num1.append("Topic " + str(i))
        for j, t in enumerate(ttm2):
            matrix[i, j] = 1 - jensen_shannon(tt, t)
    print("Compare {} with {}".format(model_pkl,model_pkl2))
    df = pd.DataFrame(matrix, columns=["Topic" + str(i) for i in range(0,len(ttm2))], index=topic_num1)
    if sorted:
        if axis == "column":
            sort_arg = np.argsort(matrix, axis=0)[::-1]
            sorted_v = np.sort(matrix, axis=0)[::-1]
        elif axis == "row":
            sort_arg = np.argsort(matrix, axis=1)[::-1]
            sorted_v = np.sort(matrix, axis=1)[::-1]
        else:
            raise Exception("Invalid axis")
        result = []
        num_rows, num_columns = matrix.shape
        for c in range(0, num_columns):
            c_list = []
            for r in range(0, num_rows):
                t = "Topic " + str(sort_arg[r, c])
                v = round(sorted_v[r, c], 3)
                c_list.append((t, v))
            result.append(c_list)
        topic_label = ["Topic " + str(i) for i in range(0, num_columns)]
        df2 = pd.DataFrame(result, index=topic_label)
        return df2
    else:
        return df

def get_entropy(model_pkl):
    """
    Calcuate the entropy for a all topics in a Topic Model

    :param model_pkl: string to load a model
    :return: list with entropies per topic
    """
    topicmodel = TopicModel.load(model_pkl)
    ttm = topicmodel.get_topic_token_matrix()
    res = []
    for i in range(0,len(ttm)):
        row = ttm[i,:]
        entropy_per_row = round(entropy(row),3)
        res.append(entropy_per_row)
    return res

def plot_entropy(model_pkl, title, limit_x, sorted = True):
    """
    The entropies per topic are plotted in a graph

    :param model_pkl: string to load a model
    :param sorted: Sorts the entropies descending
    :type boolean
    :return:
    """
    entropy = get_entropy(model_pkl)
    entropy = np.array(entropy)

    #plt.title("Entropy for {}".format(model_pkl[:(len(model_pkl)) - 4]))
    plt.title("Entropy for {}".format(title))
    plt.xlabel('Topics')
    plt.ylabel('Entropy')

    if sorted:
        indices = np.argsort(entropy)[::-1]
        sorted_e = entropy[indices]
        if limit_x <= 50:
            plt.bar(range(0, len(entropy)), sorted_e, 0.3)
            plt.xticks(range(0,len(indices)), indices, rotation=90)
            plt.xlim(-1, limit_x)
        else:
            plt.xticks([], [])
            plt.fill_between(range(0, len(indices)), sorted_e)
        plt.ylim(0, max(sorted_e) + 1)
    else:
        plt.bar(range(0, len(entropy)), entropy, 0.3)
        #plt.xticks(range(0, len(entropy)),range(0, len(entropy)), rotation=45)
        # plt.savefig("test.png",bbox_inches = "tight")
    path = title.replace(" ","_")
    plt.savefig("D:\\Bachelorarbeit\\Thesis\\cleanthesis-TUM\\cleanthesis-TUM\\gfx\\Entropy\\{}{}.pdf".format(path,limit_x), bbox_inches='tight')
    plt.show()

def plot_alphas_and_entropy(alpha,tm,title):
    """

    :param alpha: list of calculated alphas
    :param tm: string pkl
    :param title: title for the plot
    :return: Plot with alpha and entropy
    """
    plt.title("Correlation between alpha and entropy for \n{}".format(title))
    plt.xlabel("Alpha")
    plt.ylabel('Entropy')
    e190 = get_entropy(tm)
    fit = np.polyfit(alpha, e190, 1)
    fit_fn = np.poly1d(fit)
    print(fit_fn)

    plt.plot(alpha, e190, 'yo', alpha, fit_fn(alpha), "black")
    plt.plot(alpha, e190, "ro")
    path = title.replace(" ", "_")
    #plt.savefig("D:\\Bachelorarbeit\\Thesis\\cleanthesis-TUM\\cleanthesis-TUM\\gfx\\Correlation\\{}.pdf".format(path),
    #            bbox_inches='tight')
if __name__ == "__main__":
    t140 = "intern_consistency/topic_models/lda_german_editorial_articles_140.pkl"
    plot_entropy(t140,"tit")