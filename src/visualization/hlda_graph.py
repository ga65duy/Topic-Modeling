from graphviz import Digraph
import src.features.vectorizer as v

"""
Vizualize the Hierarchical Topic Model
"""
def param_for_Hlda(texts):
    """
    Return the needed parameters for the hierarchical Topic Model
    :param texts:
    :type list
    :return: doc_converted (id list for the tokens)
    :return token_list
    """
    min_df = 0.005
    max_df = 0.9
    vec = v.Vectorizer("tf", texts, min_df = min_df, max_df =max_df)
    token_list = vec.get_token_list()
    vocab = vec.token2id_mapping
    doc_converted = []
    for text in texts:
        doc = text.split()
        doc_words_indicies = []
        for word in doc:
            try:
                doc_words_indicies.append(vocab[word.lower()])
            except KeyError:
                continue

        #doc = [vocab[word.lower()] for word in doc]
        #doc_converted.append(doc)
        doc_converted.append(doc_words_indicies)
    return doc_converted, token_list

def graph(hlda):
    """
    Visualize Hlda as a tree

    :param hlda: results from the hierarchical Topic model
    :return: Hlda as a graph
    """
    dict_list = hlda.dict_for_tree(5)
    dot = Digraph(comment='Topics')
    t = Digraph ("HLda")
    for l in dict_list:
        dot.node(str(l["topic_id"]),("Topic " + str(l["topic_id"])+": \n" + l["topic"].replace(",","\n")), shape = "box")
        if l["parent"] != -1:
            dot.edge(str(l["parent"]),str(l["topic_id"]),constraint='true')
    return dot