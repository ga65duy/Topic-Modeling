from graphviz import Digraph
import src.features.vectorizer as v

def param_for_Hlda(texts):
    vec = v.Vectorizer("tf", texts)
    token_list = vec.get_token_list()
    vocab = vec.token2id_mapping
    doc_converted = []
    for text in texts:
        doc = text.split()
        doc = [vocab[word.lower()] for word in doc]
        doc_converted.append(doc)
    return doc_converted, token_list

def graph(hlda):
    dict_list = hlda.dict_for_tree(5)
    dot = Digraph(comment='Topics')
    t = Digraph ("HLda")
    for l in dict_list:
        dot.node(str(l["topic_id"]),("Topic " + str(l["topic_id"])+": \n" + l["topic"].replace(",","\n")), shape = "box")
        if l["parent"] != -1:
            dot.edge(str(l["parent"]),str(l["topic_id"]),constraint='true')
    return dot