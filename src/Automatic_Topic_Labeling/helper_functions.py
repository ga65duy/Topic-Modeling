import src.data.data_loader as dl

# def min_word_length (list, min):
#     """
#     Throws out all the words of the document lists that the lenght is equal or smaller then
#     min character amount.
#
#     :param list: list list of documentens
#     :param min: minimal amount of characters (int)
#     :return: new document list list
#     """
#     new_outer_list = []
#     outer_list = list
#     for inner_list in outer_list:
#         new_inner_list = []
#         for element in inner_list:
#             if len(element) >= min:
#                 new_inner_list.append(element)
#             else:
#                 continue
#         new_outer_list.append(new_inner_list)
#     return new_outer_list
#
#
# def load_doc (language ,pos ,texttype, kind):
#     """
#     Load the documents for a specific language, type and texttpye
#
#     :param language: string "german" or "english"
#     :type string
#     :param type: "editorial" or "blogs"
#     :type: string
#     :param texttype:string "article_texts", "comment_texts" or "thread_texts"
#     :type string
#     :param kind: Specifies the kind of processing applied on the data. The possible values "tagged","with_2bigramms","wobigrams" correspond to the folders of the data directory.
#     type: string
#     :return:  list list of loaded documents
#     """
#
#     if texttype == "article_texts":
#         data = dl.get_articles_by_type(language, "editorial", kind = kind, metadata =["article_text_pos"])
#         pos_docs = data['article_text_pos']
#
#     elif texttype == "comment_texts":
#         data = dl.get_comments_by_type(language, "editorial", kind= kind)
#         pos_docs = data["comment_text_pos"]
#
#     elif texttype == "thread_texts":
#         data = dl.get_forum_threads_by_language(language, kind=kind)
#         pos_docs = data['article_text_pos'] + data["comment_text_pos"]
#
#     texts = data[texttype]
#     docs = []
#     for item in texts:
#         splitted = item.split()
#         docs.append(splitted)
#
#     if pos:
#         posTuple = []
#         for l in pos_docs:
#             docsList = []
#             for d in l:
#                 docsList.append(tuple(d))
#             posTuple.append(docsList)
#         return docs,posTuple
#     else:
#         return docs
#
#     #docs = min_word_length(docs,3)
#
#
# def load_taggs(language,type):
#     """
#     !!!!!!Prüfen
#     Load the tuples, containing the words and the pos taggs from the text
#
#     :param language:string "german" or "english"
#     :param type: "comments" or "articles"
#     :return: Tuple with the word and the Pos-tag
#     """
#
#     if type == "article_texts":
#         data = dl.get_articles_by_type(language, "editorial", kind='tagged', metadata=['article_text_pos'])
#         print(type(data))
#
#     elif type == "comment_texts":
#         data = dl.get_comments_by_type(language, "editorial", kind='tagged', metadata=["comment_text_pos"])
#
#     elif type == "thread_texts":
#         data = dl.get_forum_threads_by_language(language, kind="tagged",metadata=["article_text_pos","comment_text_pos"])
#
#     else:
#         raise Exception ('Type {} not supported.'.format(type))
#
#     #docs = data['article_text_pos'] + data["comment_text_pos"]
#
#     docsTuple = []
#     texts = data[type]
#     for l in texts:
#         docsList = []
#         for d in l:
#             docsList.append(tuple(d))
#         docsTuple.append(docsList)
#     return docsTuple

def print_label_df(tm,labels):
    """
    Combine the topics and their label in a dataframe

    :param tm: loaded topicmodel
    :param labels: generated labels in a list
    :return: dataframe
    """
    df = tm.get_topics_dataframe()
    lablelist = []
    for l in labels:
        lablelist.append(' '.join(l[0]))
    df["label"] = lablelist
    return df
