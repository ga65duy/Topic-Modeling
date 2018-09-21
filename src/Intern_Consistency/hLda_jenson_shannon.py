import numpy as np
import pandas as pd
from scipy.stats import entropy

def get_max_level(dictionary1,dictionary2):
    level1 = []
    level2 = []
    for d in dictionary1:
        level1.append(d["level"])
    for d2 in dictionary2:
        level2.append(d2["level"])
    if max(level1)== max(level2):
        return max(level1)
    elif max(level1) < max(level2):
        return max(level1)
    else:
        return max(level2)

def get_matrix_for_all_levels(dict1,dict2):
    res_id = []
    res_topic = []
    max_level = get_max_level(dict1,dict2)
    for h in range(0,max_level +1):
        dist_for_matrix1 = normalize_distributions(dict1, h)
        dist_for_matrix2 = normalize_distributions(dict2, h)
        d,x = get_dist_matrix(dist_for_matrix1, dist_for_matrix2)
        res_id.append(d)
        res_topic.append(x)
    return res_id,res_topic

def filter_dict_for_tree_level(dictionary, level):
    level_list = []
    for d in dictionary:
        if d["level"] == level:
            level_list.append(d)
    return level_list

def normalize_distributions(dictionary,level):
    searched_level_dict = filter_dict_for_tree_level(dictionary,level)
    normalized_dist = []
    for e in searched_level_dict:
        result_dict = {}
        result_dict["normalized_dist"] = (e["word_dist"]/sum(e["word_dist"]))
        result_dict["topic_id"] = e["topic_id"]
        result_dict["topic_words"] = e["topic"]
        normalized_dist.append(result_dict)
    return normalized_dist

def get_dist_matrix(dictlist1,dictlist2):
    matrix = np.zeros((len(dictlist1), len(dictlist2)))
    topic_id_list1 = []
    topic_id_list2 = []

    topic_word1 = []
    topic_word2 = []

    for i, di1 in enumerate(dictlist1):
        topic_id_list1.append("Topic " + str(di1["topic_id"]))
        topic_word1.append(di1["topic_words"])
        for j, di2 in enumerate(dictlist2):
            matrix[i,j] = jensen_shannon(di1["normalized_dist"], di2["normalized_dist"])

    for dict2 in dictlist2:
        topic_id_list2.append("Topic " + str(dict2["topic_id"]))
        topic_word2.append(dict2["topic_words"])
    df = pd.DataFrame(matrix, columns= topic_id_list2, index = topic_id_list1)
    df2 = pd.DataFrame(matrix, columns= topic_word2, index = topic_word1)
    return df, df2

def jensen_shannon(vec1, vec2, num_features=None):
    """Calculate Jensen-Shannon distance between two probability distributions using `scipy.stats.entropy`.
    Parameters
    ----------
    vec1 : {numpy.ndarray, list of (int, float)}
        Distribution vector.
    vec2 : {numpy.ndarray, list of (int, float)}
        Distribution vector.
    num_features : int, optional
        Number of features in the vectors.
    Returns
    -------
    float
        Jensen-Shannon distance between `vec1` and `vec2`.
    Notes
    -----
    This is a symmetric and finite "version" of :func:`gensim.matutils.kullback_leibler`.
    """
    avg_vec = 0.5 * (vec1 + vec2)
    return 0.5 * (entropy(vec1, avg_vec) + entropy(vec2, avg_vec))

