from nltk.corpus import wordnet as wn
import numpy as np
from itertools import combinations
from collections import defaultdict
import operator
from nltk.corpus import wordnet_ic

class Wordnet(object):

    def __init__(self):
        self.brown_ic = wordnet_ic.ic('ic-brown.dat')
        self.semcor_ic = wordnet_ic.ic('ic-semcor.dat')

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
            elif similarity_fun =="res_similarity":
                return syn1.res_similarity(syn2, self.brown_ic)
            elif similarity_fun == "jcn_similarity":
                return syn1.jcn_similarity(syn2, self.brown_ic)
            elif similarity_fun == "lin_similarity":
                return syn1.lin_similarity(syn2, self.semcor_ic)
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
                result.append(sorted_labels[:num_labels])
            else:
                result.append([h for h, s in sorted_labels][:num_labels])
        return result

if __name__ =='__main__':
    # topics = [
    #     ['plant', 'table', "chair", 'tree'],
    #     ['cat', 'dog', 'farmer']
    # ]
    # tl = Wordnet()
    # list = [['cat', 'horse', 'cow', 'farmer']]
    # tm = tm.TopicModel.load("topic_models/lda/ENED_lda_english_editorial_articles_130.pkl")
    # print(tm.get_topics())
    # lables = tl.get_topic_labels(tm.get_topics(5),"path_similarity")
    # print(tm.get_topics(5))
    # print(lables)
    import sys

    sys.path.append("../..")
    from src.Automati_Topic_Labeling_Wordnet.extrinsic_topic_labler import ExtrensicTopicLabeler
    from src.Automati_Topic_Labeling_Wordnet.wordnet_embeddings import Wordnet
    from src.Automati_Topic_Labeling_Wordnet.polyglot_embeddings import get_topic_labels as pl
    from src.models import topic_models as tm
    from src.Automati_Topic_Labeling_Wordnet.topic_embedding import words_for_topics as wt

    tm = tm.TopicModel.load("topic_models/lda/ENED_lda_english_editorial_articles_130.pkl")
    topics = tm.get_topics()
    topics_df = tm.get_topics_dataframe()
    e = ExtrensicTopicLabeler()
    labels = e.get_topic_labels(topics, values=False)
    new_topics = wt(topics, 3)
    labels_preprocessed = e.get_topic_labels(new_topics, values=False)
    ### take only the best label
    llist = []
    for ll in labels:
        llist.append(ll[0])
    llpre = []
    for ll in labels_preprocessed:
        llpre.append(ll[0])
    combined_df = e.combine_topic_and_labels(topics_df, llist)
    combined_df['preprocessed_labels'] = llpre






