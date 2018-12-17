from polyglot.mapping import Embedding
from nltk.corpus import wordnet as wn
from itertools import combinations
from collections import defaultdict
import operator
import numpy as np

# embeddings = Embedding.load("D:/Bachelorarbeit/Projekte/polyglot_data/embeddings2/en/embeddings_pkl.tar.bz2")
embeddings = Embedding.from_glove("D:/Bachelorarbeit/Projekte/tm-maria/models/word_embeddings/glove.6B/glove.6B.100d.txt")


def get_words_from_sysets(synset):
    """
    Get all the words form a synset.
    e.g [Synset('cat.n.01'), Synset('guy.n.01')] will return [cat,guy]

    :param synset: list of synsets form wordnet
    :type list
    :return: list of synonyms (synsets) for the synset
    """
    synlist = []
    for s in synset:
        syns = s.lemmas()[0].name()
        synlist.append(syns)
    return synlist


def calc_similarity_between_words(word1, word2):
    """
    Calculate the distances with word embeddings for all sysnsets from wordnet  of word1 and word2
    and return the lowest common hypernym of the two words with the smallest distance.

    :param word1: word from a synset
    :type string
    :param word2: word from a synset
    :type string
    :return: min distance between two words and their lowest common hypernym
    """
    # pos = wn.Noun is mandatory otherwise the lowest common hypernym cant be found because of part of speach
    word1_synsets = wn.synsets(word1, pos=wn.NOUN)
    word2_synsets = wn.synsets(word2, pos=wn.NOUN)

    w1 = get_words_from_sysets(word1_synsets)
    w2 = get_words_from_sysets(word2_synsets)

    sim_matrix = np.zeros((len(w1), len(w2)))

    for i in range(len(w1)):
        for j in range(len(w2)):
            try:
                sim_matrix[i, j] = embeddings.distances(w1[i], [w2[j]])
            except KeyError:
                sim_matrix[i, j] = 1000
                continue

    w1_ind, w2_ind = np.unravel_index(np.nanargmin(sim_matrix, axis=None), sim_matrix.shape)
    lowest_common_hyp = (word1_synsets[w1_ind]).lowest_common_hypernyms(word2_synsets[w2_ind])
    return (sim_matrix[w1_ind, w2_ind], lowest_common_hyp)


def get_label_for_topic(topic):
    """
    Calculate for all permutations of the top words in a topic the distance between the words (see: function  calc_similarity_between_words)
    The socres with the same lowest common hypernyms will be summed up.

    :param topic: list of words
    :type list
    :return: list of the lowest common hypernyms and the score (distance)
    """
    labels = defaultdict(float)
    perms = combinations(topic, 2)
    for (w1, w2) in perms:
        try:
            sim_score, hyp = calc_similarity_between_words(w1, w2)
            hyp_name = hyp[0].lemmas()[0].name()
            labels[hyp_name] += sim_score
        except Exception:
            continue
    return labels


def get_topic_labels(topics, num_labels=1, values=False):
    """
    Calculate the labels for all topics in a topic model

    :param topics: the topics of a Topic Model
    :type list list
    :param num_labels: number of lables for one topic
    :type int
    :param values: scores(distance) will be shown if vales = True
    :type boolean
    :return: list of labels
    """
    result = []
    for topic in topics:
        labels = get_label_for_topic(topic)
        sorted_labels = sorted(labels.items(), key=operator.itemgetter(1), reverse=False)
        if values:
            result.append(sorted_labels[:num_labels])
        else:
            result.append([h for h, s in sorted_labels][:num_labels])
    return result

if __name__ == "__main__":
    topics = [
        ['plant', 'table', "chair", 'tree'],
        ['cat', 'dog', "cow", "horse", "snake"]]
    list = ['cat', 'horse', 'cow', 'farmer']

    # embeddings = Embedding.load("D:/Bachelorarbeit/Projekte/polyglot_data/embeddings2/en/embeddings_pkl.tar.bz2")
    # embeddings = embeddings.normalize_words()

    # dist = embeddings.distances("dog",["cat"])
    # print(dist)

    # tm = tm.TopicModel.load("topic_models/lda/ENED_lda_english_editorial_articles_130.pkl")
    # lables = get_topic_labels(tm.get_topics())
    print(get_topic_labels(topics))
    # print(get_words_from_sysets(word1_synsets))
