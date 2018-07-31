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


def load_doc (language ,type ,texttype, kind):
    #thread_texts= foren, comment_texts = comments
    if texttype == "article_texts":
        data = dl.get_articles_by_type(language, type, kind = kind)

    elif texttype == "comment_texts":
        #haben blogs ausgeschlossen
        data = dl.get_comments_by_type(language, type, kind= kind)

    elif texttype == "thread_texts":
        data = dl.get_forum_threads_by_language(language, kind=kind)

    texts = data[texttype]
    docs = []
    for item in texts:
        splitted = item.split()
        docs.append(splitted)

    docs = min_word_length(docs,3)

    return docs


def get_topic_lables(tagged_docs, docs,pos ,model , vec, trained_tm, n_top_words, n_cand_lables, lable_min_df, n_labels):
    #pos=[('ADJD', 'NN'),]
    finder = BigramLabelFinder('pmi', min_freq=lable_min_df, pos=pos)

    #muss nicht anzahl der topic sein?
    if tagged_docs != []:
        cand_labels = finder.find(tagged_docs, top_n=n_cand_lables)
    else:
        cand_labels = finder.find(docs, top_n=n_cand_lables)

    #print(cand_labels)
    #vec = Vectorizer.load(load_vectorizerpath)


    pmi_cal = PMICalculator(
        doc2word_vectorizer=vec.sk_model,
        doc2label_vectorizer=LabelCountVectorizer())
    pmi_w2l = pmi_cal.from_texts(docs, cand_labels)

    print("PMI"+str(pmi_w2l))

    print("Topic modeling using " + model)
    #tm_model = TopicModel.load(model, trained_tm)
    topic_token_matrix = trained_tm.get_topic_token_matrix()

    # for i, topic_dist in enumerate(topic_token_matrix):
    #     top_word_ids = np.argsort(topic_dist)[:-n_top_words:-1]
    #     topic_words = [pmi_cal.index2word_[id_]
    #                    for id_ in top_word_ids]
    #     print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    table = trained_tm.get_topics_dataframe()
    display(table)

    ranker = LabelRanker(apply_intra_topic_coverage=False)

    labels = ranker.top_k_labels(topic_models=topic_token_matrix,
                               pmi_w2l=pmi_w2l,
                               index2label=pmi_cal.index2label_,
                               label_models=None,
                               k = n_labels)
    return labels

def load_taggs(language,type):
    data = dl.get_articles_by_type(language, type, kind='tagged', metadata=['article_text_pos'])
    docs = data['article_text_pos']

    docsTuple = []
    for l in docs:
        docsList = []
        for d in l:
            docsList.append(tuple(d))
        docsTuple.append(docsList)
    return docsTuple

if __name__ == "__main__":
    language = "german"
    type = "editorial"
    texttype = "article_texts"
    model = "lda"
    pos = [('ADJD', 'NN'),]

    vec = Vectorizer.load('tagged/vectorizer/lda_german_editorial_pos.pkl')
    tm = TopicModel.load('tagged/topic_models/lda/lda_german_editorial_tagged_articles_190.pkl')

    docs = load_doc(language, type, texttype,"tagged")

    tagged_docs = load_taggs(language,type)

    labels = get_topic_lables(tagged_docs, docs,pos,model,vec,tm,n_top_words = 15, n_cand_lables = 100,lable_min_df = 15, n_labels = 8)

    print("\nTopical labels:")
    print("-" * 20)
    for i, labels in enumerate(labels):
        print(u"Topic {}: {}\n".format(
            i,
            ', '.join(map(lambda l: ' '.join(l), labels))
        ))


