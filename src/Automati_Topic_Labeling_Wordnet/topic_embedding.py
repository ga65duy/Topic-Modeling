from src.Automati_Topic_Labeling_Wordnet.extrinsic_topic_labler import ExtrensicTopicLabeler
from src.Automati_Topic_Labeling_Wordnet.wordnet_embeddings import Wordnet
from src.Automati_Topic_Labeling_Wordnet.polyglot_embeddings import get_topic_labels as pl
#from src.Automati_Topic_Labeling_Wordnet.tree_labeling import TreeLabel as tree
from itertools import permutations, combinations
from polyglot.mapping import Embedding
from itertools import chain
from src.models import topic_models as tm


#embeddings = Embedding.load("D:/Bachelorarbeit/Projekte/polyglot_data/embeddings2/en/embeddings_pkl.tar.bz2")
embeddings = Embedding.from_glove("D:/Bachelorarbeit/Projekte/tm-maria/models/word_embeddings/glove.6B/glove.6B.100d.txt")

def topic_word_distance(word1, word2):
    try:
        dist = embeddings.distances(word1,[word2])
    except KeyError:
        return
    return ((word1,word2),dist)

def get_distances_for_topic(topic):
    perms = combinations(topic, 2)
    distlist = []
    for (w1, w2) in perms:
        try:
            words, dist = topic_word_distance(w1, w2)
            distlist.append((words,dist))
        except Exception:
            continue
    return distlist

def get_words_with_bestsim(topics,top = 5,values = False):
    distlist = get_distances_for_topic(topics)
    sortedlist = sorted(distlist, key=lambda tup: tup[1])[:top]

    if values:
        return sortedlist
    else:
        wordlist= [i[0] for i in sortedlist]
        return list(set(list(chain.from_iterable(wordlist))))

def words_for_topics(topics,top = 5,values=False):
    best_words_per_topic_list = []
    for l in topics:
        filtered_topics = get_words_with_bestsim(l,top,values)
        best_words_per_topic_list.append(filtered_topics)
    return best_words_per_topic_list


if __name__ == '__main__':
    tm = tm.TopicModel.load("topic_models/lda/ENED_lda_english_editorial_articles_130.pkl")
    topics = (tm.get_topics())
    print(topics)
    new_topics = words_for_topics(topics,3)
    print(new_topics)

    ##extrinsic
    print("extrinsic")
    e = ExtrensicTopicLabeler()
    print(e.print_topic_labels(topics))
    print("preprocessed")
    print(e.print_topic_labels(new_topics))

    print("wornetembeddings")
    wb = Wordnet()
    print(wb.get_topic_labels(topics,"path_similarity"))
    print("preprocessed")
    print(wb.get_topic_labels(new_topics,"path_similarity"))

    print("polyglotEmbeddings")
    print(pl(topics))
    print("preprocessed")
    print(pl(new_topics))