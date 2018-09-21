from src.Automatic_Topic_Labeling.lable_finder import BigramLabelFinder
from src.Automatic_Topic_Labeling.PMICalculator import PMICalculator
from src.Automatic_Topic_Labeling.text import LabelCountVectorizer
from src.Automatic_Topic_Labeling.label_ranker import LabelRanker
from IPython.display import display
import src.Automatic_Topic_Labeling.helper_functions as hp

import src.data.data_loader as dl
from src.features.vectorizer import Vectorizer
from src.models.topic_models import TopicModel



def get_topic_lables(tagged_docs, docs,pos ,model , vec, trained_tm, n_top_words, n_cand_lables, lable_min_df, n_labels):
    #pos=[('ADJD', 'NN'),]
    finder = BigramLabelFinder('pmi', min_freq=lable_min_df, pos=pos)

    #muss nicht anzahl der topic sein?
    if pos:
        cand_labels = finder.find(tagged_docs, top_n=n_cand_lables)
    else:
        cand_labels = finder.find(docs, top_n=n_cand_lables)

    #print(cand_labels)
    #vec = Vectorizer.load(load_vectorizerpath)


    pmi_cal = PMICalculator(
        doc2word_vectorizer=vec.sk_model,
        doc2label_vectorizer=LabelCountVectorizer())
    pmi_w2l = pmi_cal.from_texts(docs, cand_labels)

    # print("PMI"+str(pmi_w2l))
    #
    # print("Topic modeling using " + model)
    #tm_model = TopicModel.load(model, trained_tm)
    topic_token_matrix = trained_tm.get_topic_token_matrix()

    # for i, topic_dist in enumerate(topic_token_matrix):
    #     top_word_ids = np.argsort(topic_dist)[:-n_top_words:-1]
    #     topic_words = [pmi_cal.index2word_[id_]
    #                    for id_ in top_word_ids]
    #     print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    table = trained_tm.get_topics_dataframe()
    #display(table)

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
    pos = [('ADJD', 'NN'),]

    vec = Vectorizer.load('tagged/vectorizer/lda_german_editorial_pos.pkl')
    tm = TopicModel.load('tagged/topic_models/lda/lda_german_editorial_tagged_articles_190.pkl')

    docs = hp.load_doc(language, type, texttype,"tagged")

    tagged_docs = hp.load_taggs(language,type)

    labels = get_topic_lables(tagged_docs, docs,pos,model,vec,tm,n_top_words = 15, n_cand_lables = 100,lable_min_df = 5, n_labels = 8)

    print("\nTopical labels:")
    print("-" * 20)
    for i, labels in enumerate(labels):
        print(u"Topic {}: {}\n".format(
            i,
            ', '.join(map(lambda l: ' '.join(l), labels))
        ))


