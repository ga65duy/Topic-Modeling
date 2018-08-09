import polyglot
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
from itertools import permutations, combinations
from collections import defaultdict
import operator
from src.models import topic_models as tm

class TreeLabel(object):

    #topics: list
    def calc_similarity_between_synsets(self, similarity_fun, syn1, syn2):
        if similarity_fun == "path_similarity":
            return syn1.path_similarity(syn2)
        elif similarity_fun =="lch_similarity":
            return syn1.lch_similarity(syn2)
        else:
            raise Exception ("similarity"+similarity_fun+" not supported")

    def calc_similarity_between_words(self, similarity_fun, word1, word2):
        '''

        :param similarity_fun:
        :param word1:
        :param word2:
        :return: the max similarity of the synset of two words and their lowest common hypernym
        '''
        word1_synsets = wn.synsets(word1, pos= wn.NOUN)
        word2_synsets = wn.synsets(word2,pos= wn.NOUN)
        #print(word1_synsets,word2_synsets)
        if word1_synsets == [] or word2_synsets == []:
            raise Exception("Empty list")
        else:

            num_word1_meanings = len(word1_synsets)
            num_word2_meanings = len(word2_synsets)
            sim_matrix = np.zeros((num_word1_meanings, num_word2_meanings))
            for i in range(num_word1_meanings):
                for j in range(num_word2_meanings):
                    sim_matrix[i,j] = self.calc_similarity_between_synsets(similarity_fun,word1_synsets[i],word2_synsets[j])
            #matrixposition of max value
            w1_ind, w2_ind = np.unravel_index(np.nanargmax(sim_matrix, axis=None), sim_matrix.shape)
            lowest_common_hyp = (word1_synsets[w1_ind]).lowest_common_hypernyms(word2_synsets[w2_ind])

            hyper = lambda s: s.hypernyms()
            hirarchyList = []
            hirarchyList.append(lowest_common_hyp[0])
            hirarchyList.extend(lowest_common_hyp[0].closure(hyper))

        return sim_matrix[w1_ind, w2_ind], hirarchyList

    def get_label_for_topic(self, topic, similarity_fun):
        '''

        :param topic:
        :param similarity_fun:
        :return: dictionary with the hypernyms and the summd up score for same hypernyms
        '''
        res = []
        perms = combinations(topic, 2)
        #for p in perms:
            #print(p)
        for (w1, w2) in perms:
            try:
                sim_score, hyp = self.calc_similarity_between_words(similarity_fun, w1,w2)
                res.append((sim_score,hyp))
            except Exception:
                continue
        return res

    def get_topic_labels(self, topics, similarity_fun, num_labels=1, values=False):
        '''

        :param topics:
        :param similarity_fun:
        :param num_labels:
        :param values:
        :return: a sorted list list with the amount of lables per topic (with or without the similarity score)
        '''
        result = []
        for topic in topics:
            labels = self.get_label_for_topic(topic, similarity_fun)
            result.append(labels)
        return result

    # def make_similariy_matrix(self,topic,similarity_fun):
    #     topic_words_ln= len(topic)
    #     matrix = pd.DataFrame(columns= [topic], index = [topic])
    #     for i in range(topic_words_ln):
    #         for j in range(topic_words_ln):
    #             matrix.iloc[i,j] = self.calc_similarity_between_words(similarity_fun,topic[i],topic[j])
    #     return matrix


if __name__ =='__main__':
    w = TreeLabel()
    res = w.calc_similarity_between_words("path_similarity","dog","cat")
    print(res)

    tm = tm.TopicModel.load("topic_models/lda/ENED_lda_english_editorial_articles_130.pkl")
    print(tm.get_topics()[:1])
    lables = w.get_topic_labels(tm.get_topics()[:1], "path_similarity")
    #print(tm.get_topics(5))
    for l in lables:
        for n in l:
            print(n)


