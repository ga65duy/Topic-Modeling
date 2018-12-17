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
