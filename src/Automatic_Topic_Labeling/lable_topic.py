from src.Automatic_Topic_Labeling.lable_finder import BigramLabelFinder
from src.Automatic_Topic_Labeling.PMICalculator import PMICalculator
from src.Automatic_Topic_Labeling.text import LabelCountVectorizer
from src.Automatic_Topic_Labeling.label_ranker import LabelRanker
"""
Reference:
---------------------
Qiaozhu Mei, Xuehua Shen, Chengxiang Zhai,
Automatic Labeling of Multinomial Topic Models, 2007

implementation was taken from "https://github.com/xiaohan2012/chowmein/tree/master/chowmein" and have been adopted to our data.
Following we changed in the given implementation:
    We used our vectoriter, our preprocessing and applied POS-Tagging and stored it in the json, too.
    Additionally, we prefiltered our datasets and throw out all words which had a smaller length then 3.
"""
def get_topic_lables(tagged_docs, docs, pos, vec, trained_tm, n_cand_lables, lable_min_df, n_labels):
    finder = BigramLabelFinder('pmi', min_freq=lable_min_df, pos=pos)

    if pos:
        cand_labels = finder.find(tagged_docs, top_n=n_cand_lables)
    else:
        cand_labels = finder.find(docs, top_n=n_cand_lables)

    pmi_cal = PMICalculator(doc2word_vectorizer=vec.sk_model, doc2label_vectorizer=LabelCountVectorizer())
    pmi_w2l = pmi_cal.from_texts(docs, cand_labels)

    topic_token_matrix = trained_tm.get_topic_token_matrix()

    table = trained_tm.get_topics_dataframe()

    ranker = LabelRanker(apply_intra_topic_coverage=False)

    labels = ranker.top_k_labels(topic_models=topic_token_matrix,
                                 pmi_w2l=pmi_w2l,
                                 index2label=pmi_cal.index2label_,
                                 label_models=None,
                                 k=n_labels)
    return labels


