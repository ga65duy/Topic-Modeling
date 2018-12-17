from nltk.corpus import wordnet as wn
import operator


class ExtrensicTopicLabeler(object):
    '''
    Generate labels for the topic models with wordnet.
    '''

    def get_hypernyms_for_single_word(self, word):
        """
        Get all the hypernyms for the synsets of a given word of a certain topic

        :param word: a word from a topic
        :type string
        :return: set of hypernyms for the given word
        """
        result = []
        # synsets = wn.synsets(word, pos = wn.NOUN)
        synsets = wn.synsets(word)
        for synset in synsets:
            hypernyms = synset.hypernyms()
            for hyp in hypernyms:
                result.append(hyp.lemmas()[0].name())
        return set(result)

    def get_hypernyms_for_word_list(self, words):
        """
        All hypernyms with the score (if some words have a common hypernym, add up their sore and multiply it with the amount of words which have the same hypernym)
        for the word list in a dictionary.

        :param words: list of words
        :type list
        :return: a dictionary with the hypernym as keys and their score
        """
        results = []
        for word in words:
            hypernyms = self.get_hypernyms_for_single_word(word)
            for hypernym in hypernyms:
                result = list(filter(lambda h_dict: h_dict['hyp'] == hypernym, results))
                if result:
                    h_dict = result[0]
                    h_dict['score'].append(1 / len(hypernyms))
                else:
                    results.append({'hyp': hypernym, 'score': [1 / len(hypernyms)]})
        results_dict = {}
        for hyp_dict in results:
            mul = len(hyp_dict['score'])
            hyp_dict['score'] = sum(hyp_dict['score']) * mul
            results_dict[hyp_dict['hyp']] = hyp_dict['score']
        return results_dict

    def get_topic_labels(self, topics, values=True, num=5):
        """
        Get the top labels for every topic.

        :param topics: all topics from a topic model
        :type list
        :param values: show the score for a hypernym if values = True
        :typy boolean
        :param num: number of top labels for one topic
        :type int
        :return: list of top labels for every topic in a topic model
        """
        result = []
        for topic in topics:
            # hypname, score dict
            hypernyms = self.get_hypernyms_for_word_list(topic)
            # tuplilist(hypname,doublescorevalue)
            sorted_hypernyms = sorted(hypernyms.items(), key=operator.itemgetter(1), reverse=True)
            if values:
                result.append(sorted_hypernyms[:num])
            else:
                result.append([h for h, s in sorted_hypernyms][:num])
        return result

    def combine_topic_and_labels(self, topics_df, lables):
        """
        Combines the topics with the lables in one dataframe

        :param topics_df: topic dataframe
        :type dataframe
        :param lables: labels for the topics
        :type list
        :return: dataframe with the topics and the generated labels
        """
        assert (len(topics_df) == len(lables))
        # assert(len(lables[0]) == 1)
        topics_df['labels'] = lables
        return topics_df


if __name__ == '__main__':
    e = ExtrensicTopicLabeler()
    print(e.get_hypernyms_for_single_word('cat'))
    #tm = tm.TopicModel.load("topic_models/lda/ENED_lda_english_editorial_articles_130.pkl")
    #topics = tm.get_topics()
    #print(e.get_topic_labels(topics))
