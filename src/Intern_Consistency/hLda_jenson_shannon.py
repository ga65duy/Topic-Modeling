import numpy as np
import pandas as pd
from scipy.stats import entropy

"""
Calculate the Similatities for the same levels of a tree with Jenson Shannon for the Hierarchical Topic Models

"""
def get_max_common_level(dictionary1, dictionary2):
    """
    Calculate the maximal common level of 2 trees

    :param dictionary1: list of dictionaries with following keys: 'topic_id', 'level', 'parent', 'topic' and 'word_dist' for a tree
    :param dictionary2: list of dictionaries for the second tree
    :return: the maximal common level of 2 trees (int)
    """

    level1 = []
    level2 = []
    for d in dictionary1:
        level1.append(d["level"])
    for d2 in dictionary2:
        level2.append(d2["level"])
    if max(level1) == max(level2):
        return max(level1)
    elif max(level1) < max(level2):
        return max(level1)
    else:
        return max(level2)

def get_matrix_for_all_common_levels(dict1, dict2):
    """
    Calculate the similarity for the same levels in two different trees

    :param dict1: list of dictionaries with following keys: 'topic_id', 'level', 'parent', 'topic' and 'word_dist' for a tree
    :param dict2: list of dictionaries for the second tree
    :return: the maximal common level of 2 trees (int)
    :return:
        :param res_id: list of the similarites and topic ids
        :param red_topic: list of similarities and top words of the topics
    """
    res_id = []
    res_topic = []
    max_level = get_max_common_level(dict1, dict2)
    for h in range(0, max_level + 1):
        dist_for_matrix1 = normalize_distributions(dict1, h)
        dist_for_matrix2 = normalize_distributions(dict2, h)
        d, x = get_dist_matrix(dist_for_matrix1, dist_for_matrix2)
        res_id.append(d)
        res_topic.append(x)
    return res_id, res_topic


def filter_dict_for_tree_level(dictionary, level):
    """
    Filer the dictionary for a certain level

    :param dictionary: list of dictionaries with following keys: 'topic_id', 'level', 'parent', 'topic' and 'word_dist' for a tree
    :param level: int
    :return: list of dictionaries with the searched level
    """
    level_list = []
    for d in dictionary:
        if d["level"] == level:
            level_list.append(d)
    return level_list


def normalize_distributions(dictionary, level):
    """
    Normalize the word distribution
    :param dictionary: list of dictionaries with following keys: 'topic_id', 'level', 'parent', 'topic' and 'word_dist' for a tree
    :param level: int
    :return: list of dictionaries with the keys "normalize_dist", "topic-id" and "topic_words"
    """
    searched_level_dict = filter_dict_for_tree_level(dictionary, level)
    normalized_dist = []
    for e in searched_level_dict:
        result_dict = {}
        result_dict["normalized_dist"] = (e["word_dist"] / sum(e["word_dist"]))
        result_dict["topic_id"] = e["topic_id"]
        result_dict["topic_words"] = e["topic"]
        normalized_dist.append(result_dict)
    return normalized_dist


def get_dist_matrix(dictlist1, dictlist2):
    """
    Generate 2datafrmaes with the similarities for a certain level.
    One is shown with the topic ids and the other one with top words of a topic

    :param dictlist1: dictionaries with the keys "normalize_dist", "topic-id" and "topic_words"
    :param dictlist2: dictionaries with the keys "normalize_dist", "topic-id" and "topic_words"
    :return:
        :param df: Dataframe with the similarities and the topic ids
        :param df2: Dataframe with the similarities and the top words of a topic
    """
    matrix = np.zeros((len(dictlist1), len(dictlist2)))
    topic_id_list1 = []
    topic_id_list2 = []

    topic_word1 = []
    topic_word2 = []

    for i, di1 in enumerate(dictlist1):
        topic_id_list1.append("Topic " + str(di1["topic_id"]))
        topic_word1.append(di1["topic_words"])
        for j, di2 in enumerate(dictlist2):
            matrix[i, j] = 1 - (jensen_shannon(di1["normalized_dist"], di2["normalized_dist"]))

    for dict2 in dictlist2:
        topic_id_list2.append("Topic " + str(dict2["topic_id"]))
        topic_word2.append(dict2["topic_words"])
    df = pd.DataFrame(matrix, columns=topic_id_list2, index=topic_id_list1)
    df2 = pd.DataFrame(matrix, columns=topic_word2, index=topic_word1)
    return df, df2


def jensen_shannon(vec1, vec2):
    """Calculate Jensen-Shannon distance between two probability distributions using `scipy.stats.entropy`.

    :param vec1: list
    :param vec2 : list
    :return Jensen-Shannon distance between `vec1` and `vec2`.

    """
    avg_vec = 0.5 * (vec1 + vec2)
    return round(0.5 * (entropy(vec1, avg_vec) + entropy(vec2, avg_vec)), 3)
