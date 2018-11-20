import polyglot
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
from itertools import permutations, combinations
from collections import defaultdict
import operator
from src.models import topic_models as tm

class Wordnet(object):

    def calc_similarity_between_synsets(self, similarity_fun, syn1, syn2):
        """
        Calculate the similarities with the similarityfunctions from wordnet

        :param similarity_fun: path_similarity(range between 0 and 1, 1 represents identity) or lch_similarity
        :type string
        :param syn1: synset for a word
        :param syn2: synsets for a word
        :type wordnet.Synset
        :return: similarity of the two word-synsets, value between 0 and 1
        """
        if similarity_fun == "path_similarity":
            return syn1.path_similarity(syn2)
        elif similarity_fun =="lch_similarity":
            return syn1.lch_similarity(syn2)
        else:
            raise Exception ("similarity"+similarity_fun+" not supported")

    def calc_similarity_between_words(self, similarity_fun, word1, word2):
        '''
        Calculate the similarity with similarityfunction from wordnet for all sysnsets of word1 and word2
        and return the lowest common hypernym of the two words with the highest similarity.

        :param similarity_fun: path_similarity or lch_similarity
        :type string
        :param word1: string
        :param word2: string
        :return: the max similarity of the synset of two words and their lowest common hypernym
        '''
        word1_synsets = wn.synsets(word1, pos= wn.NOUN)
        word2_synsets = wn.synsets(word2,pos= wn.NOUN)

        if word1_synsets == [] or word2_synsets == []:
            raise Exception("Empty list")
        else:
            num_word1_meanings = len(word1_synsets)
            num_word2_meanings = len(word2_synsets)
            sim_matrix = np.zeros((num_word1_meanings, num_word2_meanings))
            for i in range(num_word1_meanings):
                for j in range(num_word2_meanings):
                    sim_matrix[i,j] = self.calc_similarity_between_synsets(similarity_fun,word1_synsets[i],word2_synsets[j])

            w1_ind, w2_ind = np.unravel_index(np.nanargmax(sim_matrix, axis=None), sim_matrix.shape)
            lowest_common_hyp = (word1_synsets[w1_ind]).lowest_common_hypernyms(word2_synsets[w2_ind])

        return sim_matrix[w1_ind, w2_ind], lowest_common_hyp

    def get_label_for_topic(self, topic, similarity_fun):
        '''
        Get the labels for one topic.

        :param topic: list
        :param similarity_fun: see above
        :return: dictionary with the hypernyms and the summd up score for same hypernyms
        '''
        labels = defaultdict(float)
        perms = combinations(topic, 2)
        for (w1, w2) in perms:
            try:
                sim_score, hyp = self.calc_similarity_between_words(similarity_fun, w1,w2)
                hyp_name = hyp[0].lemmas()[0].name()
                labels[hyp_name] += sim_score
            except Exception:
                continue
        return labels

    def get_topic_labels(self, topics, similarity_fun, num_labels=1, values=False):
        '''
        Get the topics for all topics in a topic model

        :param topics: list
        :param similarity_fun: see above
        :param num_labels: number of labels per topic
        :type int
        :param values: scores will be shown if vales = True
        :type boolean
        :return: a sorted list list with the amount of lables per topic (with or without the similarity score)
        '''
        result = []
        for topic in topics:
            labels = self.get_label_for_topic(topic, similarity_fun)
            sorted_labels = sorted(labels.items(), key=operator.itemgetter(1), reverse=True)
            if values:
                result.append( sorted_labels[:num_labels])
            else:
                result.append([h for h, s in sorted_labels][:num_labels])
        return result

if __name__ =='__main__':
    topics = [
        ['plant', 'table', "chair", 'tree'],
        ['cat', 'dog', 'farmer']
    ]
    tl = Wordnet()
    list = [['cat', 'horse', 'cow', 'farmer']]
    tm = tm.TopicModel.load("topic_models/lda/ENED_lda_english_editorial_articles_130.pkl")
    print(tm.get_topics())
    lables = tl.get_topic_labels(tm.get_topics(5),"path_similarity")
    print(tm.get_topics(5))
    print(lables)



