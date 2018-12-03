import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

"""
Plot the labels and their occurence for topics
"""


def wordcount(df, name):
    """
    wordcounts with labels for labeling with similarity functions
    :param df: df with topics and labels
    :param name: "labels" or "preprocessed_labels"
    :return: dictionary with labels and label counts
    """
    label = list(df[name])
    label_list = []
    for l in label:
        if not l:
            label_list.append("Empty")
        else:
            label_list.append(l[0])

    return Counter(label_list)


def wordcount_scor(df, name):
    """
    wordcounts with labels for labeling with scoringfunction
    :param df: df with topics and labels
    :param name: "labels" or "preprocessed_labels"
    :return: dictionary with labels and label counts
    """
    label = list(df[name])
    label_list = []
    for l in label:
        if not l:
            label_list.append("Empty")
        else:
            label_list.append(l)

    return Counter(label_list)


def selected_words(dict, select_word_list):
    """
    select words  and word counts from dictionary
    :param dict:
    :param select_word_list:
    :return: list
    """
    selected_words = []
    for w in select_word_list:
        if w in dict:
            selected_words.append((w,dict[w]))
    return selected_words


def plot_label_counts(df, name, filename, fun, title,sort=False, cut_ones=False):
    """
    Plot the number of different labels

    :param df:
    :param name: name of coolumn
    :param filename:
    :param fun: function wordcount_scor or wordcount
    :return:
    """
    counter = fun(df, name)
    labels, values = zip(*counter.items())
    indices = np.arange(len(labels))

    plt.title('Label counts for {}'.format(title))
    plt.xlabel('Label')
    plt.ylabel('Number of labels')

    if sort:
        sorted_values_idx = np.argsort(values)[::-1]
        labels = np.asarray(labels)[sorted_values_idx]
        values = np.asarray(values)[sorted_values_idx]

        if cut_ones:
            one_idx = np.argwhere(values==1)[0][0]
            labels = labels[0:one_idx+2]
            values = values[0:one_idx+2]
            labels[one_idx+1] = '...'
            indices = np.arange(len(labels))

    plt.yticks(range(0,max(values),2))
    plt.bar(indices, values)
    plt.xticks(indices, labels, rotation=90)
    plt.autoscale(True)
    plt.savefig("C:\\Users\\Maria\\Desktop\\{}.pdf".format(filename),bbox_inches='tight')
    plt.show()
