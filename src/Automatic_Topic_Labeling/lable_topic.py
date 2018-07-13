from src.Automatic_Topic_Labeling.lable_finder import BigramLabelFinder
from src.Automatic_Topic_Labeling.PMICalculator import PMICalculator
from src.Automatic_Topic_Labeling.text import LabelCountVectorizer
from src.Automatic_Topic_Labeling.label_ranker import LabelRanker
from IPython.display import display
import numpy as np

import src.data.data_loader as dl
from src.features.vectorizer import Vectorizer
from src.models.topic_models import TopicModel

def min_word_length (list, min):
    new_outer_list = []
    outer_list = list
    for inner_list in outer_list:
        new_inner_list = []
        for element in inner_list:
            if len(element) >= min:
                new_inner_list.append(element)
            else:
                continue
        new_outer_list.append(new_inner_list)
    return new_outer_list


def load_doc (language ,type ,texttype):
    #thread_texts= foren, comment_texts = comments
    if texttype == "article_texts":
        data = dl.get_articles_by_type(language, type, kind = "wobigrams")

    elif texttype == "comment_texts":
        #haben blogs ausgeschlossen
        data = dl.get_comments_by_type(language, type, kind="wobigrams")

    elif texttype == "thread_texts":
        data = dl.get_forum_threads_by_language(language, kind="wobigrams")

    texts = data[texttype]
    docs = []
    for item in texts:
        splitted = item.split()
        docs.append(splitted)

    docs = min_word_length(docs,3)

    return docs


def get_topic_lables(docs , model , load_vectorizerpath, load_tm_path,n_top_words, n_cand_lables, lable_min_df, n_labels):
    finder = BigramLabelFinder('pmi', min_freq=lable_min_df, pos=[])

    #muss nicht anzahl der topic sein?
    cand_labels = finder.find(docs, top_n=n_cand_lables)

    #print(cand_labels)
    vec = Vectorizer.load(load_vectorizerpath)


    pmi_cal = PMICalculator(
        doc2word_vectorizer=vec.sk_model,
        doc2label_vectorizer=LabelCountVectorizer())
    pmi_w2l = pmi_cal.from_texts(docs, cand_labels)

    print("PMI"+str(pmi_w2l))

    print("Topic modeling using " + model)
    tm_model = TopicModel.load(model, load_tm_path)
    topic_token_matrix = tm_model.get_topic_token_matrix()

    # for i, topic_dist in enumerate(topic_token_matrix):
    #     top_word_ids = np.argsort(topic_dist)[:-n_top_words:-1]
    #     topic_words = [pmi_cal.index2word_[id_]
    #                    for id_ in top_word_ids]
    #     print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    table = tm_model.get_topics_dataframe()
    display(table)

    ranker = LabelRanker(apply_intra_topic_coverage=False)

    labels = ranker.top_k_labels(topic_models=topic_token_matrix,
                               pmi_w2l=pmi_w2l,
                               index2label=pmi_cal.index2label_,
                               label_models=None,
                               k = n_labels)
    return labels

if __name__ == "__main__":
    language = "german"
    type = "editorial"
    texttype = "article_texts"
    model = "lda"

    load_vectorizer = "lda_german_editorial_without_bigrams.pkl"
    load_tm_path = "lda_german_editorial_without_bigrams_articles_190.pkl"

    docs = load_doc(language, type, texttype)

    labels = get_topic_lables(docs,model,load_vectorizer,load_tm_path,n_top_words = 15, n_cand_lables = 100,lable_min_df = 15, n_labels = 8)

    print("\nTopical labels:")
    print("-" * 20)
    for i, labels in enumerate(labels):
        print(u"Topic {}: {}\n".format(
            i,
            ', '.join(map(lambda l: ' '.join(l), labels))
        ))


