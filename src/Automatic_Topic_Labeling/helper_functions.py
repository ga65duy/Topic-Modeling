import src.data.data_loader as dl

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

